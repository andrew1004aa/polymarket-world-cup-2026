#!/usr/bin/env node
/** Build the 104-event FIFA/Polymarket mapping from existing local sources. */
import fs from "node:fs/promises";
import path from "node:path";
import * as XLSX from "xlsx";

const ROOT = process.cwd();
const FIFA_SOURCE = path.resolve(process.argv[2] ?? "source/world-cup_2026.xlsx");
const RAW_MARKETS = path.join(ROOT, "raw/markets/markets.csv");
const FINAL_MARKETS = path.join(ROOT, "intermediate/market_partitions/markets_final.csv");
const EVENT_DIR = path.join(ROOT, "raw/events");
const SOURCE_DIR = path.join(EVENT_DIR, "source");
const EVENTS_CSV = path.join(EVENT_DIR, "events.csv");
const MAPPING_CSV = path.join(EVENT_DIR, "event_market_mapping.csv");
const QC_JSON = path.join(EVENT_DIR, "event_table_qc.json");

function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') { field += '"'; i += 1; }
      else if (char === '"') quoted = false;
      else field += char;
    } else if (char === '"') quoted = true;
    else if (char === ',') { row.push(field); field = ""; }
    else if (char === '\n') { row.push(field.replace(/\r$/, "")); rows.push(row); row = []; field = ""; }
    else field += char;
  }
  if (field.length || row.length) { row.push(field.replace(/\r$/, "")); rows.push(row); }
  const headers = rows.shift();
  return rows.filter(r => r.some(v => v !== "")).map(values =>
    Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]))
  );
}

function escapeCsv(value) {
  const text = value == null ? "" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

async function writeCsv(filePath, headers, records) {
  const lines = [headers.map(escapeCsv).join(",")];
  for (const record of records) lines.push(headers.map(h => escapeCsv(record[h])).join(","));
  const temporary = `${filePath}.${process.pid}.tmp`;
  await fs.writeFile(temporary, `${lines.join("\n")}\n`, "utf8");
  await fs.rename(temporary, filePath);
}

function normalizeTeam(value) {
  const aliases = new Map([
    ["bosnia & herzegovina", "bosnia and herzegovina"],
    ["cape verde islands", "cabo verde"],
    ["congo dr", "dr congo"],
    ["iran", "ir iran"],
    ["ivory coast", "côte d'ivoire"],
    ["south korea", "korea republic"],
    ["usa", "united states"],
  ]);
  const clean = String(value).trim().toLowerCase().replaceAll("’", "'").replace(/\s+/g, " ");
  return aliases.get(clean) ?? clean;
}

function stageForMatch(matchId) {
  if (matchId <= 72) return "Group stage";
  if (matchId <= 88) return "Round of 32";
  if (matchId <= 96) return "Round of 16";
  if (matchId <= 100) return "Quarter-final";
  if (matchId <= 102) return "Semi-final";
  if (matchId === 103) return "Third-place match";
  return "Final";
}

function bstAndUtc(date, time) {
  const bst = `${date}T${time}:00+01:00`;
  const parsed = new Date(bst);
  if (Number.isNaN(parsed.getTime())) throw new Error(`Invalid BST fixture timestamp: ${date} ${time}`);
  return { bst, utc: parsed.toISOString().replace(".000Z", "Z") };
}

function winnerTeam(question) {
  const match = /^Will (.+) win on \d{4}-\d{2}-\d{2}\?$/.exec(question);
  if (!match) throw new Error(`Unable to parse winner question: ${question}`);
  return match[1];
}

function sortObjectKeys(value) {
  return Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)));
}

await fs.mkdir(EVENT_DIR, { recursive: true });
await fs.mkdir(SOURCE_DIR, { recursive: true });
await fs.copyFile(FIFA_SOURCE, path.join(SOURCE_DIR, "fifa_world_cup_2026.xlsx"));

const fifaWorkbook = XLSX.readFile(FIFA_SOURCE, { cellDates: false });
const fifaSheet = fifaWorkbook.Sheets["world-cup"];
if (!fifaSheet) throw new Error("Worksheet 'world-cup' was not found in the FIFA file");
const fifaValues = XLSX.utils.sheet_to_json(fifaSheet, { header: 1, raw: false, defval: "" });
const expectedFifaHeaders = ["Date", "Time", "Home Team", "Away Team", "Result"];
if (JSON.stringify(fifaValues[0]) !== JSON.stringify(expectedFifaHeaders)) {
  throw new Error(`Unexpected FIFA headers: ${JSON.stringify(fifaValues[0])}`);
}
const fixtures = fifaValues.slice(1).map((row, index) => ({
  fifa_match_id: index + 1,
  match_date_bst: String(row[0]).trim(),
  match_time_bst: String(row[1]).trim(),
  home_team: String(row[2]).trim(),
  away_team: String(row[3]).trim(),
  result: String(row[4]).trim(),
}));
if (fixtures.length !== 104) throw new Error(`Expected 104 FIFA fixtures; found ${fixtures.length}`);

const rawMarkets = parseCsv(await fs.readFile(RAW_MARKETS, "utf8")).filter(r => r.market_type === "match");
const finalMarkets = parseCsv(await fs.readFile(FINAL_MARKETS, "utf8")).filter(r => r.market_type === "match");
if (rawMarkets.length !== 312 || finalMarkets.length !== 312) {
  throw new Error(`Expected 312 raw and final match markets; found ${rawMarkets.length} and ${finalMarkets.length}`);
}
const finalByMarketId = new Map(finalMarkets.map(row => [row.market_id, row]));
const eventGroups = new Map();
for (const market of rawMarkets) {
  const group = eventGroups.get(market.event_id) ?? [];
  group.push({ ...market, final: finalByMarketId.get(market.market_id) });
  eventGroups.set(market.event_id, group);
}
if (eventGroups.size !== 104) throw new Error(`Expected 104 Polymarket events; found ${eventGroups.size}`);

const preparedGroups = [...eventGroups.entries()].map(([polymarketEventId, markets]) => {
  if (markets.length !== 3 || markets.some(m => !m.final)) {
    throw new Error(`Event ${polymarketEventId} does not contain three fully joined markets`);
  }
  const gameIds = new Set(markets.map(m => m.game_id));
  const eventSlugs = new Set(markets.map(m => m.final.event_slug));
  const kickoffUtc = new Set(markets.map(m => m.end_date));
  if (gameIds.size !== 1 || eventSlugs.size !== 1 || kickoffUtc.size !== 1) {
    throw new Error(`Inconsistent identifiers within event ${polymarketEventId}`);
  }
  const draw = markets.filter(m => m.final.market_subtype === "Match draw");
  const winners = markets.filter(m => m.final.market_subtype === "Match winner");
  if (draw.length !== 1 || winners.length !== 2) throw new Error(`Invalid market roles for event ${polymarketEventId}`);
  return {
    polymarketEventId,
    gameId: [...gameIds][0],
    eventSlug: [...eventSlugs][0],
    kickoffUtc: [...kickoffUtc][0],
    draw: draw[0],
    winners: winners.map(m => ({ market: m, team: winnerTeam(m.question) })),
    used: false,
  };
});

const events = [];
const mapping = [];
const unmatchedFixtures = [];
const kickoffMetadataDifferences = [];
for (const fixture of fixtures) {
  const times = bstAndUtc(fixture.match_date_bst, fixture.match_time_bst);
  const wantedTeams = new Set([normalizeTeam(fixture.home_team), normalizeTeam(fixture.away_team)]);
  const teamCandidates = preparedGroups.filter(group => {
    const groupTeams = new Set(group.winners.map(w => normalizeTeam(w.team)));
    return !group.used && groupTeams.size === 2 &&
      [...wantedTeams].every(team => groupTeams.has(team));
  });
  const exactCandidates = teamCandidates.filter(group =>
    new Date(group.kickoffUtc).toISOString().replace(".000Z", "Z") === times.utc
  );
  // Team pairs identify the event; exact kickoff agreement is a QC check.
  // This handles a documented one-hour Polymarket metadata difference for
  // Mexico–England while preserving both source timestamps in the output.
  const candidates = exactCandidates.length === 1 ? exactCandidates : teamCandidates;
  if (candidates.length !== 1) {
    unmatchedFixtures.push({ ...fixture, scheduled_kickoff_utc: times.utc, candidate_count: candidates.length });
    continue;
  }
  const group = candidates[0];
  group.used = true;
  const polymarketMarketEndUtc = new Date(group.kickoffUtc).toISOString().replace(".000Z", "Z");
  const kickoffDifferenceSeconds = Math.round(
    (new Date(polymarketMarketEndUtc).getTime() - new Date(times.utc).getTime()) / 1000
  );
  const kickoffDelayMinutes = Math.max(0, Math.round(-kickoffDifferenceSeconds / 60));
  const kickoffDelayReason = fixture.fifa_match_id === 92 && kickoffDelayMinutes === 60 ? "weather" : "";
  if (kickoffDifferenceSeconds !== 0) {
    kickoffMetadataDifferences.push({
      fifa_match_id: fixture.fifa_match_id,
      match_name: `${fixture.home_team} vs. ${fixture.away_team}`,
      fifa_actual_kickoff_utc: times.utc,
      polymarket_market_end_utc: polymarketMarketEndUtc,
      difference_seconds: kickoffDifferenceSeconds,
    });
  }
  const home = group.winners.find(w => normalizeTeam(w.team) === normalizeTeam(fixture.home_team));
  const away = group.winners.find(w => normalizeTeam(w.team) === normalizeTeam(fixture.away_team));
  if (!home || !away) throw new Error(`Unable to assign home/away roles for FIFA match ${fixture.fifa_match_id}`);

  const event = {
    event_id: fixture.fifa_match_id,
    fifa_match_id: fixture.fifa_match_id,
    polymarket_event_id: group.polymarketEventId,
    polymarket_game_id: group.gameId,
    event_slug: group.eventSlug,
    stage: stageForMatch(fixture.fifa_match_id),
    match_date_bst: fixture.match_date_bst,
    match_time_bst: fixture.match_time_bst,
    actual_kickoff_bst: times.bst,
    actual_kickoff_utc: times.utc,
    scheduled_kickoff_utc: polymarketMarketEndUtc,
    polymarket_market_end_utc: polymarketMarketEndUtc,
    kickoff_metadata_difference_seconds: kickoffDifferenceSeconds,
    kickoff_delay_minutes: kickoffDelayMinutes,
    kickoff_delay_reason: kickoffDelayReason,
    timezone: "Europe/London",
    home_team: fixture.home_team,
    away_team: fixture.away_team,
    match_name: `${fixture.home_team} vs. ${fixture.away_team}`,
    result: fixture.result,
    market_count: 3,
    home_win_market_id: home.market.market_id,
    home_win_condition_id: home.market.condition_id,
    home_win_question: home.market.question,
    away_win_market_id: away.market.market_id,
    away_win_condition_id: away.market.condition_id,
    away_win_question: away.market.question,
    draw_market_id: group.draw.market_id,
    draw_condition_id: group.draw.condition_id,
    draw_question: group.draw.question,
    fixture_source: "FIFA official world-cup_2026.xlsx",
    market_source: "Polymarket public APIs / Dune aggregate metadata",
  };
  events.push(event);

  for (const [role, item] of [["home_win", home.market], ["away_win", away.market], ["draw", group.draw]]) {
    mapping.push({
      event_id: event.event_id,
      fifa_match_id: event.fifa_match_id,
      polymarket_event_id: event.polymarket_event_id,
      polymarket_game_id: event.polymarket_game_id,
      event_slug: event.event_slug,
      actual_kickoff_utc: event.actual_kickoff_utc,
      home_team: event.home_team,
      away_team: event.away_team,
      market_role: role,
      market_id: item.market_id,
      condition_id: item.condition_id,
      market_subtype: item.final.market_subtype,
      question: item.question,
      resolution_status: item.final.resolution_status,
      resolved_outcome: item.final.resolved_outcome,
      yes_outcome_won: item.final.yes_outcome_won,
      resolved_on_timestamp: item.final.resolved_on_timestamp,
    });
  }
}

const unusedGroups = preparedGroups.filter(group => !group.used).map(group => group.polymarketEventId);
const eventHeaders = Object.keys(events[0] ?? {});
const mappingHeaders = Object.keys(mapping[0] ?? {});
const allMarketIds = mapping.map(row => row.market_id);
const allConditionIds = mapping.map(row => row.condition_id.toLowerCase());
const eventRoleCounts = Object.fromEntries(events.map(event => [event.event_id,
  mapping.filter(row => row.event_id === event.event_id).map(row => row.market_role).sort()
]));
const expectedRoles = JSON.stringify(["away_win", "draw", "home_win"]);
const roleErrors = Object.entries(eventRoleCounts).filter(([, roles]) => JSON.stringify(roles) !== expectedRoles);
const errors = [];
if (events.length !== 104) errors.push(`Expected 104 events; found ${events.length}`);
if (mapping.length !== 312) errors.push(`Expected 312 mappings; found ${mapping.length}`);
if (unmatchedFixtures.length) errors.push(`${unmatchedFixtures.length} FIFA fixtures were unmatched`);
if (unusedGroups.length) errors.push(`${unusedGroups.length} Polymarket event groups were unused`);
if (new Set(allMarketIds).size !== mapping.length) errors.push("Market IDs are not unique");
if (new Set(allConditionIds).size !== mapping.length) errors.push("Condition IDs are not unique");
if (roleErrors.length) errors.push(`${roleErrors.length} events do not have exactly three roles`);
if (mapping.some(row => !/^0x[0-9a-f]{64}$/i.test(row.condition_id))) errors.push("Invalid condition ID format found");
if (events.some(event => event.market_count !== 3)) errors.push("Invalid event market_count found");

await writeCsv(EVENTS_CSV, eventHeaders, events);
await writeCsv(MAPPING_CSV, mappingHeaders, mapping);

const qc = {
  generated_at: new Date().toISOString(),
  fifa_fixture_rows: fixtures.length,
  raw_match_markets: rawMarkets.length,
  final_match_markets: finalMarkets.length,
  polymarket_event_groups: eventGroups.size,
  output_events: events.length,
  output_event_market_rows: mapping.length,
  unique_event_ids: new Set(events.map(row => row.event_id)).size,
  unique_polymarket_event_ids: new Set(events.map(row => row.polymarket_event_id)).size,
  unique_market_ids: new Set(allMarketIds).size,
  unique_condition_ids: new Set(allConditionIds).size,
  home_win_rows: mapping.filter(row => row.market_role === "home_win").length,
  away_win_rows: mapping.filter(row => row.market_role === "away_win").length,
  draw_rows: mapping.filter(row => row.market_role === "draw").length,
  unmatched_fixtures: unmatchedFixtures,
  unused_polymarket_event_groups: unusedGroups,
  role_errors: roleErrors,
  kickoff_metadata_differences: kickoffMetadataDifferences,
  timezone: "Europe/London",
  bst_utc_offset_during_sample: "+01:00",
  errors,
  notes: [
    "FIFA Excel is the authoritative fixture and result source.",
    "Polymarket event_id/game_id and the three market identifiers come from existing public-API market files.",
    "FIFA actual BST kickoff timestamps were converted to UTC by subtracting one hour.",
    "Polymarket market end is retained as scheduled kickoff; FIFA time is retained as actual kickoff.",
    "FIFA match 92 started 60 minutes late because of weather, as documented by the researcher.",
    "No statistical analysis, inference, or market classification was performed.",
  ],
};
const qcTemporary = `${QC_JSON}.${process.pid}.tmp`;
await fs.writeFile(qcTemporary, `${JSON.stringify(qc, null, 2)}\n`, "utf8");
await fs.rename(qcTemporary, QC_JSON);
if (errors.length) throw new Error(`Event-table QC failed: ${errors.join("; ")}`);

console.log(JSON.stringify(sortObjectKeys({
  events_csv: EVENTS_CSV,
  mapping_csv: MAPPING_CSV,
  qc_json: QC_JSON,
  ...qc,
}), null, 2));

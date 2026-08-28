#!/usr/bin/env ruby

# Reproducible IFTE0008-method main-text count when texcount is unavailable.
# Included: the seven main chapters, including headings and hypothesis prose.
# Excluded by construction: title/declaration/front matter/abstract, floats,
# equations, footnotes, code, references, and appendices.

sections = %w[
  introduction literature_review data research_design results discussion conclusion
]

total = 0
sections.each do |name|
  text = File.read(File.join("overleaf", "sections", "#{name}.tex"))
  text = text.gsub(/%.*$/, " ")
  text = text.gsub(/\\begin\{(?:table|figure|equation|align)\*?\}.*?\\end\{(?:table|figure|equation|align)\*?\}/m, " ")
  text = text.gsub(/\\footnote\{.*?\}/m, " ")
  text = text.gsub(/\\input\{[^}]+\}/, " ")
  text = text.gsub(/\$.*?\$/m, " ")
  text = text.gsub(/\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?/, ' \1 ')
  text = text.gsub(/[{}\\&_^~]/, " ")
  count = text.scan(/[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*/).length
  total += count
  puts format("%-20s %6d", name, count)
end
puts format("%-20s %6d", "TOTAL", total)

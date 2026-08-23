#!/usr/bin/env ruby

# Approximate, reproducible section word count when texcount is unavailable.
# Excludes comments, floats, equations, LaTeX commands, and \input tables.

sections = %w[
  introduction literature_review data research_design results discussion conclusion
]

total = 0
sections.each do |name|
  text = File.read(File.join("overleaf", "sections", "#{name}.tex"))
  text = text.gsub(/%.*$/, " ")
  text = text.gsub(/\\begin\{(?:table|figure|equation|align|description)\}.*?\\end\{(?:table|figure|equation|align|description)\}/m, " ")
  text = text.gsub(/\\input\{[^}]+\}/, " ")
  text = text.gsub(/\$.*?\$/m, " ")
  text = text.gsub(/\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?/, ' \1 ')
  text = text.gsub(/[{}\\&_^~]/, " ")
  count = text.scan(/[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*/).length
  total += count
  puts format("%-20s %6d", name, count)
end
puts format("%-20s %6d", "TOTAL", total)

import React, { useState } from 'react';
import katex from 'katex';
import { ChevronDown, ChevronRight, HelpCircle, Calculator, BookOpen, Layers } from 'lucide-react';

interface MathTerm {
  symbol: string;
  name: string;
  meaning: string;
}

interface WorkedExample {
  description: string;
  values: Record<string, string | number>;
  result: string;
}

interface ExpandableMathProps {
  equation: string;
  displayMode?: boolean;
  explanation: string;
  example?: WorkedExample;
  terms?: MathTerm[];
}

function renderLatex(latex: string, displayMode: boolean = false): string {
  try {
    return katex.renderToString(latex, {
      throwOnError: false,
      displayMode,
    });
  } catch (e) {
    return `$${latex}$`;
  }
}

interface AccordionItemProps {
  icon: React.ReactNode;
  title: string;
  isOpen: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}

function AccordionItem({ icon, title, isOpen, onToggle, children }: AccordionItemProps) {
  return (
    <div className="border-t border-zinc-800 first:border-t-0">
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-zinc-800/50 transition-colors cursor-pointer"
      >
        <span className="text-indigo-400">{icon}</span>
        <span className="flex-1 text-xs font-mono uppercase tracking-wider text-zinc-300">
          {title}
        </span>
        <span className="text-zinc-500">
          {isOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        </span>
      </button>
      <div
        className={`overflow-hidden transition-all duration-300 ease-in-out ${
          isOpen ? 'max-h-[500px] opacity-100' : 'max-h-0 opacity-0'
        }`}
      >
        <div className="px-4 pb-4 pt-1">{children}</div>
      </div>
    </div>
  );
}

export default function ExpandableMath({
  equation,
  displayMode = true,
  explanation,
  example,
  terms,
}: ExpandableMathProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [openSections, setOpenSections] = useState<Set<string>>(new Set(['explanation']));

  const toggleSection = (section: string) => {
    setOpenSections((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(section)) {
        newSet.delete(section);
      } else {
        newSet.add(section);
      }
      return newSet;
    });
  };

  return (
    <div className="my-6 rounded-none border border-zinc-800 bg-zinc-900/50 shadow-sm overflow-hidden">
      {/* Equation Header - Always Visible */}
      <div
        className="flex items-start gap-4 p-4 cursor-pointer hover:bg-zinc-800/30 transition-colors group"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex-1">
          <div
            className="text-zinc-100 overflow-x-auto"
            dangerouslySetInnerHTML={{ __html: renderLatex(equation, displayMode) }}
          />
        </div>
        <button
          className={`flex-shrink-0 p-1.5 rounded transition-all ${
            isExpanded
              ? 'bg-indigo-500/20 text-indigo-400'
              : 'text-zinc-500 group-hover:text-indigo-400'
          }`}
        >
          <HelpCircle className="w-5 h-5" />
        </button>
      </div>

      {/* Expandable Content */}
      <div
        className={`transition-all duration-300 ease-in-out ${
          isExpanded ? 'max-h-[1000px] opacity-100' : 'max-h-0 opacity-0'
        }`}
      >
        <div className="border-t border-zinc-800 bg-zinc-950/50">
          {/* Plain English Explanation */}
          <AccordionItem
            icon={<BookOpen className="w-4 h-4" />}
            title="Plain English"
            isOpen={openSections.has('explanation')}
            onToggle={() => toggleSection('explanation')}
          >
            <p className="text-sm text-zinc-300 leading-relaxed">{explanation}</p>
          </AccordionItem>

          {/* Worked Example */}
          {example && (
            <AccordionItem
              icon={<Calculator className="w-4 h-4" />}
              title="Worked Example"
              isOpen={openSections.has('example')}
              onToggle={() => toggleSection('example')}
            >
              <div className="space-y-3">
                <p className="text-sm text-zinc-400 italic">{example.description}</p>
                <div className="flex flex-wrap gap-3">
                  {Object.entries(example.values).map(([key, value]) => (
                    <div
                      key={key}
                      className="flex items-center gap-2 px-3 py-1.5 bg-zinc-800/50 border border-zinc-700 rounded-sm"
                    >
                      <span
                        className="text-indigo-400 text-sm"
                        dangerouslySetInnerHTML={{ __html: renderLatex(key, false) }}
                      />
                      <span className="text-zinc-500">=</span>
                      <span className="text-zinc-200 font-mono text-sm">{String(value)}</span>
                    </div>
                  ))}
                </div>
                <div className="mt-3 p-3 bg-zinc-800/30 border border-zinc-700/50 rounded-sm">
                  <div className="text-xs font-mono uppercase tracking-wider text-zinc-500 mb-2">
                    Result
                  </div>
                  <div
                    className="text-zinc-100"
                    dangerouslySetInnerHTML={{ __html: renderLatex(example.result, true) }}
                  />
                </div>
              </div>
            </AccordionItem>
          )}

          {/* Terms Breakdown */}
          {terms && terms.length > 0 && (
            <AccordionItem
              icon={<Layers className="w-4 h-4" />}
              title="Terms Breakdown"
              isOpen={openSections.has('terms')}
              onToggle={() => toggleSection('terms')}
            >
              <div className="space-y-3">
                {terms.map((term, idx) => (
                  <div
                    key={idx}
                    className="flex items-start gap-4 p-3 bg-zinc-800/30 border border-zinc-700/50 rounded-sm"
                  >
                    <div
                      className="flex-shrink-0 min-w-[60px] text-center text-indigo-400"
                      dangerouslySetInnerHTML={{ __html: renderLatex(term.symbol, false) }}
                    />
                    <div className="flex-1">
                      <div className="text-sm font-medium text-zinc-200 mb-1">{term.name}</div>
                      <div className="text-xs text-zinc-400">{term.meaning}</div>
                    </div>
                  </div>
                ))}
              </div>
            </AccordionItem>
          )}
        </div>
      </div>
    </div>
  );
}

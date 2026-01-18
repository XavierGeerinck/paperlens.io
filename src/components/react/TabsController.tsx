import { useEffect, useState } from 'react';

interface TabsControllerProps {
  tabs: Array<{ id: string; label: string }>;
  defaultTab?: string;
}

export default function TabsController({ tabs, defaultTab }: TabsControllerProps) {
  const [activeTab, setActiveTab] = useState(defaultTab || tabs[0]?.id || '');

  useEffect(() => {
    // Show/hide tab panels based on active tab
    tabs.forEach((tab) => {
      const panel = document.getElementById(`tab-panel-${tab.id}`);
      if (panel) {
        panel.style.display = activeTab === tab.id ? 'block' : 'none';
      }
    });
  }, [activeTab, tabs]);

  return (
    <div className="border-b border-zinc-800 mb-8">
      <div className="flex gap-1">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 font-mono text-sm uppercase tracking-wider transition-colors ${
              activeTab === tab.id
                ? 'border-b-2 border-indigo-500 text-indigo-400'
                : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
    </div>
  );
}

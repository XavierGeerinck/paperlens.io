import React from 'react';
import { MOONSHOT_IDEAS } from '../constants';
import IdeaCard from '../components/IdeaCard';
import { Sparkles, Activity, Cpu, Fingerprint, BrainCircuit, Lightbulb, ExternalLink } from 'lucide-react';
import { useUI } from '../context/UIContext';
import GameOfLife from '../components/GameOfLife';
import ScrambleText from '../components/ScrambleText';
import { SketchArrowRight, SketchCircle } from '../components/SketchElements';

const Home: React.FC = () => {
  const featuredIdea = MOONSHOT_IDEAS.find(i => i.featured);
  const otherIdeas = MOONSHOT_IDEAS.filter(i => !i.featured);
  const { openContact } = useUI();

  return (
    <div className="space-y-20 relative">
      {/* Dynamic Background */}
      <div className="fixed inset-0 z-[-1]">
        <GameOfLife />
      </div>

      {/* Identity Manifesto */}
      <section className="relative py-12 sm:py-24 border-b border-zinc-800/50 backdrop-blur-[2px]">
        <div className="flex flex-col lg:flex-row gap-12 items-start justify-between">
          <div className="max-w-3xl relative">
             <div className="flex items-center gap-3 mb-8">
                <span className="w-2 h-2 bg-white rounded-full" />
                <h2 className="text-sm font-mono text-zinc-400 uppercase tracking-widest">Personal Research Log</h2>
             </div>
             
             <h1 className="text-5xl sm:text-7xl font-bold tracking-tighter text-white mb-24 leading-[0.9]">
                THIS IS MY <br />
                <span className="text-indigo-500 relative inline-block group">
                  <ScrambleText text="DIGITAL CORTEX." />
                  <div className="absolute top-full right-0 mt-4 rotate-[-4deg] hidden sm:block pointer-events-none whitespace-nowrap z-10">
                     <span className="font-sketch text-zinc-500 text-3xl opacity-80 group-hover:opacity-100 transition-opacity">*(messy & raw)</span>
                  </div>
                </span>
             </h1>
             
             <div className="relative">
                <p className="text-xl md:text-2xl text-zinc-400 max-w-2xl font-light leading-relaxed mb-8">
                    I don't just write code; I simulate futures. This is where I document my attempts to break the status quo, one moonshot at a time.
                </p>
                {/* Handwriting Note */}
                <div className="hidden lg:block absolute right-0 top-0 transform translate-x-full rotate-12 w-48">
                    <SketchArrowRight color="#71717a" className="w-12 h-12 transform rotate-90 -ml-4" />
                    <p className="font-sketch text-xl text-zinc-500 leading-tight mt-2">
                        These are mostly drafts. Don't judge the code quality yet!
                    </p>
                </div>
             </div>

             <div className="flex flex-wrap gap-4">
                <div className="px-4 py-2 border border-zinc-700 bg-zinc-950/50 backdrop-blur-md rounded-full flex items-center gap-2 text-zinc-300 font-mono text-xs uppercase tracking-wider hover:border-indigo-500 transition-colors cursor-default group">
                  <BrainCircuit className="w-4 h-4 text-indigo-400 group-hover:animate-pulse" />
                  Neural Arch
                </div>
                <div className="px-4 py-2 border border-zinc-700 bg-zinc-950/50 backdrop-blur-md rounded-full flex items-center gap-2 text-zinc-300 font-mono text-xs uppercase tracking-wider hover:border-indigo-500 transition-colors cursor-default group">
                  <Fingerprint className="w-4 h-4 text-indigo-400 group-hover:animate-pulse" />
                  Identity
                </div>
                <div className="px-4 py-2 border border-zinc-700 bg-zinc-950/50 backdrop-blur-md rounded-full flex items-center gap-2 text-zinc-300 font-mono text-xs uppercase tracking-wider hover:border-indigo-500 transition-colors cursor-default group">
                  <Lightbulb className="w-4 h-4 text-indigo-400 group-hover:animate-pulse" />
                  Moonshots
                </div>
             </div>
          </div>

          {/* Quick Bio Card - Sketchified */}
          <div className="w-full lg:w-80 shrink-0 bg-[#0c0c0e] border-2 border-zinc-800 p-6 rotate-1 hover:rotate-0 transition-transform duration-300 shadow-[8px_8px_0px_0px_rgba(24,24,27,0.5)]">
             <div className="w-20 h-20 bg-indigo-500 rounded-full mb-4 flex items-center justify-center text-3xl font-bold text-white border-4 border-zinc-950 overflow-hidden shadow-inner relative group">
               <span className="relative z-10 font-sketch text-4xl">XG</span>
               <div className="absolute inset-0 bg-indigo-400 opacity-0 group-hover:opacity-100 transition-opacity animate-pulse z-0" />
             </div>
             <h3 className="text-white font-bold text-lg mb-1 font-space">Xavier Geerinck</h3>
             <div className="text-xl font-sketch text-indigo-400 mb-4 transform -rotate-2">Innovation & Cloud Architect</div>
             <p className="text-zinc-400 text-sm leading-relaxed mb-6 font-mono">
               Passionate about solving complex problems. Writing about Code, Cloud, and the Future at xaviergeerinck.com.
             </p>
             <div className="space-y-2">
               <button 
                 onClick={openContact}
                 className="w-full py-2 bg-white text-black font-bold text-xs uppercase tracking-widest hover:bg-zinc-200 transition-colors border border-transparent"
               >
                 Connect
               </button>
               <a 
                 href="https://xaviergeerinck.com" 
                 target="_blank" 
                 rel="noreferrer"
                 className="w-full py-2 flex items-center justify-center gap-2 border border-zinc-700 text-zinc-300 font-bold text-xs uppercase tracking-widest hover:bg-zinc-800 transition-colors"
               >
                 Visit Blog <ExternalLink className="w-3 h-3" />
               </a>
             </div>
          </div>
        </div>
      </section>

      {/* Featured Section */}
      {featuredIdea && (
        <section className="relative z-10">
          <div className="flex items-center gap-4 mb-6">
             <div className="h-px bg-indigo-500 w-12" />
             <h3 className="text-indigo-400 font-mono text-sm uppercase tracking-widest font-bold">Current Obsession</h3>
          </div>
          <IdeaCard idea={featuredIdea} variant="featured" />
        </section>
      )}

      {/* The Archive */}
      <section className="relative z-10">
        <div className="flex items-center justify-between mb-8 pb-4 border-b border-zinc-800">
           <h3 className="text-3xl font-bold text-white flex items-center gap-3 font-sketch">
             <div className="transform rotate-12 text-zinc-600">
                <Cpu className="w-8 h-8" />
             </div>
             The Idea Archive
           </h3>
           <span className="font-mono text-xs text-zinc-500 uppercase tracking-widest border border-zinc-800 px-2 py-1 rounded">
             Total Entries: {MOONSHOT_IDEAS.length}
           </span>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {otherIdeas.map((idea) => (
            <IdeaCard key={idea.id} idea={idea} />
          ))}
        </div>
      </section>

      {/* Personal CTA */}
      <section className="py-24 text-center border-t border-zinc-800 border-dashed mt-12 relative z-10 bg-zinc-950/50 backdrop-blur-sm">
        <h2 className="text-4xl font-bold text-white mb-6 font-sketch transform -rotate-1">Think my ideas are crazy?</h2>
        <p className="text-zinc-500 mb-8 max-w-lg mx-auto text-lg">Good. That means I'm on the right track. Let's debate them.</p>
        <button 
          onClick={openContact}
          className="inline-flex items-center gap-3 bg-indigo-600 text-white px-8 py-4 font-bold tracking-tight hover:bg-indigo-500 transition-colors rounded-sm shadow-[4px_4px_0px_0px_rgba(255,255,255,0.1)] group hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-none"
        >
           <Fingerprint className="w-5 h-5 group-hover:scale-110 transition-transform" />
           SEND ENCRYPTED MESSAGE
        </button>
      </section>
    </div>
  );
};

export default Home;
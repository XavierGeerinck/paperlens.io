import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowUpRight, Activity, Database, Cpu, Star } from 'lucide-react';
import { Idea } from '../types';
import { SketchCircle, SketchUnderline } from './SketchElements';

interface IdeaCardProps {
  idea: Idea;
  variant?: 'standard' | 'featured';
}

const IdeaCard: React.FC<IdeaCardProps> = ({ idea, variant = 'standard' }) => {
  const statusColor = {
    'CONCEPT': 'text-blue-400',
    'PROTOTYPE': 'text-amber-400',
    'ALPHA': 'text-green-400',
    'ARCHIVED': 'text-zinc-500',
  }[idea.status];

  const statusBorderColor = {
    'CONCEPT': '#60a5fa',
    'PROTOTYPE': '#fbbf24',
    'ALPHA': '#4ade80',
    'ARCHIVED': '#71717a',
  }[idea.status];

  if (variant === 'featured') {
    return (
      <Link to={`/idea/${idea.id}`} className="group relative block w-full">
        {/* Paper texture overlay */}
        <div className="flex flex-col md:flex-row bg-[#0c0c0e] border border-zinc-800 transition-all duration-500 overflow-hidden min-h-[400px]">
          
          {/* Image Section */}
          <div className="w-full md:w-2/5 relative overflow-hidden border-r border-zinc-800">
             <div className="absolute inset-0 bg-indigo-900/10 z-10 mix-blend-overlay" />
             <img 
              src={idea.coverImage} 
              alt={idea.title} 
              className="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-1000 grayscale group-hover:grayscale-0 opacity-80"
            />
             {/* Draft Overlay */}
             <div className="absolute top-4 left-4 z-20 transform -rotate-6">
                <SketchCircle color="#ffffff">
                    <span className="font-sketch text-2xl font-bold text-white px-2">Featured</span>
                </SketchCircle>
             </div>
          </div>

          {/* Content Section */}
          <div className="w-full md:w-3/5 p-6 md:p-12 flex flex-col justify-center relative z-30">
            <div className="absolute top-6 right-8 opacity-20 transform rotate-12">
               <span className="font-sketch text-6xl text-white">#01</span>
            </div>

            <div className="flex items-center gap-3 mb-6">
               <span className="font-sketch text-xl text-zinc-500">
                 Current Status:
               </span>
               <span className={`font-sketch text-2xl font-bold ${statusColor} transform -rotate-2`}>
                 {idea.status}
               </span>
            </div>

            <h3 className="text-3xl md:text-5xl font-bold text-white mb-6 group-hover:text-indigo-400 transition-colors leading-none tracking-tight">
              {idea.title}
            </h3>
            
            <p className="text-lg text-zinc-400 font-light leading-relaxed mb-8 max-w-xl font-sketch text-2xl">
              {idea.subtitle}
            </p>

            <div className="grid grid-cols-2 gap-8 mb-8">
               <div>
                  <div className="text-[10px] font-mono text-zinc-500 uppercase tracking-wider mb-1">Impact Analysis</div>
                  <div className="text-white font-mono relative inline-block">
                    {idea.impact}
                    <SketchUnderline color="#6366f1" className="opacity-50" />
                  </div>
               </div>
               <div>
                  <div className="text-[10px] font-mono text-zinc-500 uppercase tracking-wider mb-1">Read Time</div>
                  <div className="text-white font-mono">{idea.readTime}</div>
               </div>
            </div>

            <div className="flex items-center gap-2 mt-auto">
               <span className="text-xl font-sketch text-indigo-400 group-hover:underline decoration-indigo-500/50 underline-offset-4">
                 Read full draft
               </span>
               <ArrowUpRight className="w-4 h-4 text-indigo-400 group-hover:translate-x-1 group-hover:-translate-y-1 transition-transform" />
            </div>
          </div>
        </div>
      </Link>
    );
  }

  // Standard Card
  return (
    <Link to={`/idea/${idea.id}`} className="group relative flex flex-col h-full">
      <div className="flex flex-col h-full bg-[#0c0c0e] border border-zinc-800 hover:border-zinc-500 transition-all duration-300 relative overflow-hidden">
        
        {/* Status Header - Sketch Style */}
        <div className="flex items-center justify-between p-4 border-b border-zinc-800/50">
          <div className="transform -rotate-2">
              <SketchCircle color={statusBorderColor}>
                <span className={`px-3 py-1 font-sketch text-lg font-bold ${statusColor}`}>
                    {idea.status}
                </span>
              </SketchCircle>
          </div>
          <span className="text-[10px] font-mono text-zinc-600 uppercase">
            ID: {idea.id.substring(0, 8)}
          </span>
        </div>

        {/* Content Body */}
        <div className="p-6 flex flex-col flex-grow z-10">
          <h3 className="text-xl font-bold text-zinc-100 mb-3 group-hover:text-indigo-400 transition-colors leading-tight">
            {idea.title}
          </h3>
          
          <p className="text-zinc-400 text-lg leading-relaxed mb-6 font-sketch line-clamp-3">
            {idea.subtitle}
          </p>

          <div className="mt-auto pt-4 border-t border-dashed border-zinc-800">
             <div className="flex flex-col gap-1">
                <span className="text-[10px] font-mono text-zinc-500 uppercase tracking-wider">Estimated Impact</span>
                <span className="text-sm text-zinc-300 font-medium flex items-center gap-2">
                   {idea.impact}
                </span>
             </div>
          </div>
        </div>

        {/* Footer/Action */}
        <div className="p-4 bg-zinc-900/20 flex items-center justify-between group-hover:bg-zinc-900/40 transition-colors">
          <div className="flex gap-2">
             {idea.tags.slice(0, 2).map(tag => (
               <span key={tag} className="text-sm text-zinc-500 font-sketch">#{tag}</span>
             ))}
          </div>
          <ArrowUpRight className="w-4 h-4 text-zinc-600 group-hover:text-white transition-colors" />
        </div>
      </div>
    </Link>
  );
};

export default IdeaCard;
import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Atom, Terminal, Activity, ArrowRight, Radio } from 'lucide-react';
import { useUI } from '../context/UIContext';

const Header: React.FC = () => {
  const location = useLocation();
  const isHome = location.pathname === '/';
  const { openContact } = useUI();

  return (
    <header className="sticky top-0 z-50 w-full backdrop-blur-md bg-zinc-950/80 border-b border-zinc-800">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-3 group">
          <div className="bg-white text-black p-1.5 rounded-none group-hover:bg-indigo-500 group-hover:text-white transition-colors">
            <Atom className="w-5 h-5" />
          </div>
          <div className="flex flex-col leading-none">
            <span className="font-bold text-lg tracking-widest uppercase font-mono text-white group-hover:text-indigo-400 transition-colors">
              MY.LAB
            </span>
            <span className="text-[10px] text-zinc-500 font-mono tracking-widest uppercase">
              Personal Archive
            </span>
          </div>
        </Link>

        {isHome ? (
           <div className="hidden md:flex items-center gap-2 px-3 py-1 bg-zinc-900 border border-zinc-800 rounded-full">
              <div className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
              <span className="text-xs font-mono text-zinc-400 uppercase">Brain Activity: High</span>
           </div>
        ) : (
          <nav className="hidden md:flex items-center gap-6">
             <Link to="/" className="flex items-center gap-2 text-xs font-mono text-zinc-400 hover:text-white transition-colors uppercase tracking-wider">
               <Terminal className="w-4 h-4" />
               Index
             </Link>
          </nav>
        )}

        <div className="flex items-center gap-4">
          <button 
            onClick={openContact}
            className="hidden sm:flex items-center gap-2 text-xs font-mono text-zinc-500 hover:text-white transition-colors uppercase border border-zinc-800 px-3 py-1.5 rounded hover:border-zinc-600 bg-zinc-900/50"
          >
            <Radio className="w-3 h-3" />
            Connect
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;
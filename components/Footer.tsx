import React from 'react';
import { Github, Twitter, Linkedin, Mail, Globe } from 'lucide-react';
import { useUI } from '../context/UIContext';

const Footer: React.FC = () => {
  const { openContact } = useUI();

  return (
    <footer className="border-t border-zinc-800 bg-zinc-950 py-12 mt-12 relative z-50">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          
          <div className="text-center md:text-left">
             <div className="font-bold text-white tracking-widest uppercase font-mono mb-2">Xavier Geerinck</div>
             <p className="text-zinc-500 text-sm mb-4">
              © {new Date().getFullYear()} Moonshot Labs. All rights reserved.
            </p>
             {/* Easter Egg Hint */}
             <div className="text-[10px] font-mono text-zinc-500/20 hover:text-indigo-500/80 transition-all duration-500 cursor-help select-none tracking-widest" title="System Override Sequence">
               Running process: ↑↑↓↓←→←→BA
             </div>
          </div>

          <div className="flex items-center gap-6">
            <a href="https://xaviergeerinck.com" target="_blank" rel="noreferrer" className="text-zinc-500 hover:text-white transition-colors" title="Blog">
              <Globe className="w-5 h-5" />
            </a>
            <a href="https://github.com/xaviergeerinck" target="_blank" rel="noreferrer" className="text-zinc-500 hover:text-white transition-colors" title="GitHub">
              <Github className="w-5 h-5" />
            </a>
            <a href="https://twitter.com/XavierGeerinck" target="_blank" rel="noreferrer" className="text-zinc-500 hover:text-white transition-colors" title="Twitter">
              <Twitter className="w-5 h-5" />
            </a>
            <a href="https://www.linkedin.com/in/xaviergeerinck" target="_blank" rel="noreferrer" className="text-zinc-500 hover:text-white transition-colors" title="LinkedIn">
              <Linkedin className="w-5 h-5" />
            </a>
            <button onClick={openContact} className="text-zinc-500 hover:text-white transition-colors" title="Contact">
              <Mail className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
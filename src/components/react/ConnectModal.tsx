import { useState } from 'react';
import { X, Mail, Copy, Send, Linkedin, Globe, Github, Terminal } from 'lucide-react';

export default function ConnectModal() {
  const [isOpen, setIsOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const copyEmail = () => {
    navigator.clipboard.writeText('xavier@m18x.com');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <>
      {/* Trigger Button */}
      <button
        onClick={() => setIsOpen(true)}
        className="flex items-center gap-2 px-3 py-1.5 bg-zinc-900 border border-zinc-700 text-zinc-400 hover:text-white hover:border-zinc-500 transition-all text-[10px] font-mono uppercase tracking-widest cursor-pointer"
      >
        <Terminal className="w-3 h-3" />
        <span className="hidden sm:inline">CONNECT</span>
      </button>

      {/* Modal Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={() => setIsOpen(false)}
        >
          {/* Modal Content */}
          <div
            className="relative w-full max-w-lg bg-zinc-900 border border-zinc-800 shadow-2xl animate-in fade-in zoom-in-95 duration-200"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-zinc-800 bg-zinc-950/50">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <h2 className="text-lg font-mono font-bold text-white tracking-widest uppercase">
                  Initialize Connection
                </h2>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="text-zinc-500 hover:text-white transition-colors cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Body */}
            <div className="p-6 space-y-6">
              <p className="text-zinc-400 text-sm leading-relaxed">
                I am always open to collaboration on high-impact moonshot projects.
                Connect with me on LinkedIn or check out my latest writing.
              </p>

              {/* Email Card */}
              <div className="p-4 bg-zinc-950 border border-zinc-800 flex flex-col gap-3 group hover:border-indigo-500/50 transition-colors">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-indigo-500/10 rounded">
                      <Mail className="w-5 h-5 text-indigo-400" />
                    </div>
                    <div>
                      <div className="text-xs font-mono text-zinc-500 uppercase tracking-wider">
                        Direct Message
                      </div>
                      <div className="text-white font-medium">xavier@m18x.com</div>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={copyEmail}
                      className="p-2 hover:bg-zinc-800 text-zinc-500 hover:text-white transition-colors rounded cursor-pointer"
                      title="Copy Email"
                    >
                      <Copy className="w-4 h-4" />
                    </button>
                    <a
                      href="mailto:xavier@m18x.com"
                      className="p-2 bg-white text-black hover:bg-zinc-200 transition-colors rounded"
                      title="Send Email"
                    >
                      <Send className="w-4 h-4" />
                    </a>
                  </div>
                </div>
                {copied && (
                  <div className="text-xs text-emerald-400 font-mono">
                    Email copied to clipboard!
                  </div>
                )}
              </div>

              {/* Social Links Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <a
                  href="https://www.linkedin.com/in/xaviergeerinck"
                  target="_blank"
                  rel="noreferrer"
                  className="flex flex-col items-center justify-center p-4 bg-zinc-900 border border-zinc-800 hover:bg-zinc-800 hover:border-zinc-700 transition-all group text-center"
                >
                  <div className="mb-3 text-zinc-400 group-hover:text-white transition-colors">
                    <Linkedin className="w-5 h-5" />
                  </div>
                  <div className="text-sm font-bold text-zinc-200 mb-1">LinkedIn</div>
                  <div className="text-[10px] text-zinc-500 font-mono uppercase tracking-wider">
                    Professional Network
                  </div>
                </a>

                <a
                  href="https://paperlens.io"
                  target="_blank"
                  rel="noreferrer"
                  className="flex flex-col items-center justify-center p-4 bg-zinc-900 border border-zinc-800 hover:bg-zinc-800 hover:border-zinc-700 transition-all group text-center"
                >
                  <div className="mb-3 text-zinc-400 group-hover:text-white transition-colors">
                    <Globe className="w-5 h-5" />
                  </div>
                  <div className="text-sm font-bold text-zinc-200 mb-1">Personal Blog</div>
                  <div className="text-[10px] text-zinc-500 font-mono uppercase tracking-wider">
                    Articles & Tutorials
                  </div>
                </a>

                <a
                  href="https://github.com/xaviergeerinck"
                  target="_blank"
                  rel="noreferrer"
                  className="flex flex-col items-center justify-center p-4 bg-zinc-900 border border-zinc-800 hover:bg-zinc-800 hover:border-zinc-700 transition-all group text-center"
                >
                  <div className="mb-3 text-zinc-400 group-hover:text-white transition-colors">
                    <Github className="w-5 h-5" />
                  </div>
                  <div className="text-sm font-bold text-zinc-200 mb-1">GitHub</div>
                  <div className="text-[10px] text-zinc-500 font-mono uppercase tracking-wider">
                    Codebase & PRs
                  </div>
                </a>
              </div>
            </div>

            {/* Bottom Gradient Bar */}
            <div className="h-1 w-full bg-gradient-to-r from-indigo-500 via-purple-500 to-zinc-800" />
          </div>
        </div>
      )}
    </>
  );
}

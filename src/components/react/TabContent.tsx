import type { FC, ReactNode } from 'react';

interface DocumentationTabProps {
  children: ReactNode;
}

export const DocumentationTab: FC<DocumentationTabProps> = ({ children }) => {
  return (
    <div className="prose prose-invert prose-zinc max-w-none prose-headings:font-heading prose-code:font-mono prose-pre:bg-zinc-900 prose-pre:border prose-pre:border-zinc-800">
      {children}
    </div>
  );
};

interface PDFTabProps {
  pdfUrl: string;
}

export const PDFTab: FC<PDFTabProps> = ({ pdfUrl }) => {
  return (
    <div className="flex flex-col items-start gap-4">
      <p className="text-zinc-400">View the original research paper in PDF format.</p>
      <a
        href={pdfUrl}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center gap-2 px-4 py-2 border border-zinc-700 text-zinc-300 hover:bg-zinc-800 transition-colors font-mono text-sm"
      >
        Open PDF
      </a>
    </div>
  );
};

import type { FC, ReactNode } from 'react';
import DemoView from './DemoView';
import Tabs, { type Tab } from './Tabs';
import { DocumentationTab, PDFTab, WebTab, GitHubTab } from './TabContent';

interface IdeaTabsProps {
  documentationContent: ReactNode;
  simulation?: string;
  pdfUrl?: string;
  webUrl?: string;
  githubUrl?: string;
}

const IdeaTabs: FC<IdeaTabsProps> = ({ documentationContent, simulation, pdfUrl, webUrl, githubUrl }) => {
  const tabs: Tab[] = [
    {
      id: 'paper',
      label: 'Documentation',
      content: <DocumentationTab>{documentationContent}</DocumentationTab>,
    },
  ];

  if (simulation) {
    tabs.push({
      id: 'simulation',
      label: 'Simulation',
      content: <DemoView simulationName={simulation} />,
    });
  }

  if (pdfUrl) {
    tabs.push({
      id: 'pdf',
      label: 'PDF',
      content: <PDFTab pdfUrl={pdfUrl} />,
    });
  }

  if (webUrl) {
    tabs.push({
      id: 'website',
      label: 'Website',
      content: <WebTab webUrl={webUrl} />,
    });
  }

  if (githubUrl) {
    tabs.push({
      id: 'github',
      label: 'GitHub',
      content: <GitHubTab githubUrl={githubUrl} />,
    });
  }

  return <Tabs tabs={tabs} />;
};

export default IdeaTabs;

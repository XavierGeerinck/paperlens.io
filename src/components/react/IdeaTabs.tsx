import type { FC, ReactNode } from 'react';
import DemoView from './DemoView';
import Tabs, { type Tab } from './Tabs';
import { DocumentationTab, PDFTab } from './TabContent';

interface IdeaTabsProps {
  documentationContent: ReactNode;
  simulation?: string;
  pdfUrl?: string;
}

const IdeaTabs: FC<IdeaTabsProps> = ({ documentationContent, simulation, pdfUrl }) => {
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

  return <Tabs tabs={tabs} />;
};

export default IdeaTabs;

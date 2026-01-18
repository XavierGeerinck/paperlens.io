import type { FC } from "react";
import { Activity } from "lucide-react";

interface DemoViewProps {
	simulationName?: string;
}

// Dynamically import all simulations
const modules = import.meta.glob<React.ComponentType>('./simulations/*.tsx', { eager: true, import: 'default' });

const REGISTRY: Record<string, FC> = {};

// Build registry from dynamic imports
Object.entries(modules).forEach(([path, component]) => {
	// Extract filename: ./simulations/MySim.tsx -> MySim
	const name = path.split('/').pop()?.replace(/\.tsx$/, '');
	
	if (name) {
		REGISTRY[name] = component as FC;
		
		// Optional: Support "ShortName" if filename ends with "Simulation" 
		// (e.g. BrainMimeticSimulation -> BrainMimetic) to match legacy usage
		if (name.endsWith('Simulation')) {
			const shortName = name.replace(/Simulation$/, '');
			if (shortName && !REGISTRY[shortName]) {
				REGISTRY[shortName] = component as FC;
			}
		}
	}
});

const DemoView: FC<DemoViewProps> = ({ simulationName }) => {
	const Component = simulationName ? REGISTRY[simulationName] : null;

	if (!Component) {
		return (
			<div className="flex items-center justify-center h-96 border border-zinc-800 bg-zinc-900/50 rounded-xl">
				<div className="text-center text-slate-500 font-mono">
					<Activity className="w-12 h-12 mx-auto mb-4 opacity-20" />
					<p>Interactive simulation not available for this project.</p>
				</div>
			</div>
		);
	}

	return <Component />;
};

export default DemoView;

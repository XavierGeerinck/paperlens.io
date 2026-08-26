import type { FC } from "react";

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

		// Support "ShortName" if the filename ends with "Simulation"
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
			<div className="border border-bg2 bg-bg0h rounded-lg p-6">
				<p className="text-ink2 text-sm">
					<span className="text-danger-400">error:</span> simulation{' '}
					<span className="text-amber-400">{simulationName ?? '(none)'}</span> is not in the registry.
				</p>
				<p className="text-ink4 text-xs mt-2">
					Add <span className="text-ink2">src/components/react/simulations/{simulationName ?? 'Name'}.tsx</span>{' '}
					and it loads here automatically. The written entry below is unaffected.
				</p>
			</div>
		);
	}

	return <Component />;
};

export default DemoView;

import React from "react";
import { Activity } from "lucide-react";
import BrainMimeticSimulation from "./simulations/BrainMimeticSimulation";
import DeepSeekMHCSimulation from "./simulations/DeepSeekMHCSimulation";
import DeepSeekMoESimulation from "./simulations/DeepSeekMoESimulation";
import RLVRSimulation from "./simulations/RLVRSimulation";
import RubinArchitectureSimulation from "./simulations/RubinArchitectureSimulation";
import SubQuadraticSimulation from "./simulations/SubQuadraticSimulation";
import MappingTheMindSimulation from "./simulations/MappingTheMindSimulation";
import ObjectiveVerifierSimulation from "./simulations/ObjectiveVerifierSimulation";
import AsahiM1n1Simulation from "./simulations/AsahiM1n1Simulation";
import MLASimulation from "./simulations/MLASimulation";

interface DemoViewProps {
	simulationName?: string;
}

const REGISTRY: Record<string, React.FC> = {
	BrainMimetic: BrainMimeticSimulation,
	DeepSeekMHC: DeepSeekMHCSimulation,
	DeepSeekMoE: DeepSeekMoESimulation,
	RLVR: RLVRSimulation,
	RubinArchitecture: RubinArchitectureSimulation,
	SubQuadratic: SubQuadraticSimulation,
	MappingTheMind: MappingTheMindSimulation,
	ObjectiveVerifier: ObjectiveVerifierSimulation,
	AsahiM1n1: AsahiM1n1Simulation,
	MLASimulation: MLASimulation,
};

const DemoView: React.FC<DemoViewProps> = ({ simulationName }) => {
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

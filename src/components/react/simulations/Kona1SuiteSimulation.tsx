import type { FC } from "react";
import { useState } from "react";
import { BrainCircuit, Grid3X3 } from "lucide-react";
import { SchematicButton } from "../SketchElements";
import Kona1Simulation from "./Kona1Simulation";
import Kona1BattleshipSimulation from "./Kona1BattleshipSimulation";

const Kona1SuiteSimulation: FC = () => {
	const [view, setView] = useState<"battleship" | "sudoku">("battleship");

	return (
		<div className="space-y-4">
			<div className="flex flex-wrap gap-2">
				<SchematicButton
					label="Battleship (Kona CSP)"
					icon={<BrainCircuit size={14} />}
					onClick={() => setView("battleship")}
					active={view === "battleship"}
				/>
				<SchematicButton
					label="Sudoku (Legacy)"
					icon={<Grid3X3 size={14} />}
					onClick={() => setView("sudoku")}
					active={view === "sudoku"}
				/>
			</div>

			{view === "battleship" ? <Kona1BattleshipSimulation /> : <Kona1Simulation />}
		</div>
	);
};

export default Kona1SuiteSimulation;

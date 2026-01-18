// Cost surface for heatmap
// We overlay a grid to visualize the Value Function / CostMap
const GRID_SIZE = 20;

const createCostMap = (goal: Point, obstacles: Obstacle[]) => {
	const map: number[][] = [];
	const rows = Math.ceil(CANVAS_HEIGHT / GRID_SIZE);
	const cols = Math.ceil(CANVAS_WIDTH / GRID_SIZE);

	for (let r = 0; r < rows; r++) {
		const row = [];
		for (let c = 0; c < cols; c++) {
			const x = c * GRID_SIZE + GRID_SIZE / 2;
			const y = r * GRID_SIZE + GRID_SIZE / 2;
			const point = { x, y };

			let val = distance(point, goal);
			// Penalize obstacles heavily
			for (const obs of obstacles) {
				if (distance(point, obs) < obs.radius + AGENT_RADIUS + 5) {
					val += 1000;
				}
			}
			row.push(val);
		}
		map.push(row);
	}
	return map;
};

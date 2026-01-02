import { useState, useEffect } from "react";
import { IDEAS, IdeaWithContent } from "../generated-ideas";

export const useIdeas = () => {
	const [ideas, setIdeas] = useState<IdeaWithContent[]>(IDEAS);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	// We keep the useEffect for compatibility if needed, but it's now synchronous
	useEffect(() => {
		setIdeas(IDEAS);
		setLoading(false);
	}, []);

	return { ideas, loading, error };
};

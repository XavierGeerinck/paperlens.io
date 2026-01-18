import React, { useState, useEffect, useRef } from "react";

interface ScrambleTextProps {
	text: string;
	className?: string;
	scrambleSpeed?: number;
	revealSpeed?: number;
}

const CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>?";

const ScrambleText: React.FC<ScrambleTextProps> = ({
	text,
	className = "",
	scrambleSpeed = 30,
	revealSpeed = 50,
}) => {
	const [displayText, setDisplayText] = useState(text);
	const [isHovering, setIsHovering] = useState(false);
	const intervalRef = useRef<number | null>(null);

	const scramble = () => {
		let iteration = 0;

		if (intervalRef.current) window.clearInterval(intervalRef.current);

		intervalRef.current = window.setInterval(() => {
			setDisplayText((prev) =>
				text
					.split("")
					.map((letter, index) => {
						if (index < iteration) {
							return text[index];
						}
						if (letter === " " || letter === "\n") return letter; // Preserve spaces/newlines
						return CHARS[Math.floor(Math.random() * CHARS.length)];
					})
					.join(""),
			);

			if (iteration >= text.length) {
				if (intervalRef.current) window.clearInterval(intervalRef.current);
			}

			iteration += 1 / 3; // Slow down the reveal slightly relative to the scramble
		}, scrambleSpeed);
	};

	useEffect(() => {
		// Initial scramble on mount for effect
		scramble();
		return () => {
			if (intervalRef.current) window.clearInterval(intervalRef.current);
		};
	}, [text]);

	const handleMouseEnter = () => {
		setIsHovering(true);
		scramble();
	};

	return (
		<span
			className={`inline-block cursor-default ${className}`}
			onMouseEnter={handleMouseEnter}
			onMouseLeave={() => setIsHovering(false)}
		>
			{displayText}
		</span>
	);
};

export default ScrambleText;

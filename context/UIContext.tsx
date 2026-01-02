import React, { createContext, useContext, useState, ReactNode } from "react";

interface UIContextType {
	isContactOpen: boolean;
	openContact: () => void;
	closeContact: () => void;
}

const UIContext = createContext<UIContextType | undefined>(undefined);

export const UIProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
	const [isContactOpen, setIsContactOpen] = useState(false);

	const openContact = () => setIsContactOpen(true);
	const closeContact = () => setIsContactOpen(false);

	return (
		<UIContext.Provider value={{ isContactOpen, openContact, closeContact }}>
			{children}
		</UIContext.Provider>
	);
};

export const useUI = () => {
	const context = useContext(UIContext);
	if (!context) {
		throw new Error("useUI must be used within a UIProvider");
	}
	return context;
};

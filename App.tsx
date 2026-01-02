import React, { useEffect } from "react";
import { HashRouter, Routes, Route, useLocation } from "react-router-dom";
import { HelmetProvider } from "react-helmet-async";
import Home from "./pages/Home";
import IdeaDetail from "./pages/IdeaDetail";
import Header from "./components/Header";
import Footer from "./components/Footer";
import { UIProvider } from "./context/UIContext";
import ContactModal from "./components/ContactModal";
import GameOfLife from "./components/GameOfLife";
import { USER_CONFIG } from "./config";

const ScrollToTop = () => {
	const { pathname } = useLocation();

	React.useEffect(() => {
		window.scrollTo(0, 0);
	}, [pathname]);

	return null;
};

const BackgroundLayer = () => {
	const location = useLocation();
	const isHome = location.pathname === "/";

	if (!isHome) return null;

	return (
		<div className="fixed inset-0 z-[-1] pointer-events-none">
			<GameOfLife />
			<div className="absolute inset-0 bg-zinc-950/60 bg-[radial-gradient(circle_at_center,transparent_0%,#09090b_100%)]" />
		</div>
	);
};

const App: React.FC = () => {
	useEffect(() => {
		// Geeky Console Easter Egg
		console.log(
			`                                                           
%c > SYSTEM ONLINE.
%c > Looking at the source? We should talk. 
%c > ${USER_CONFIG.email}
      `,
			"color: #6366f1; font-weight: bold; font-family: monospace;",
			"color: #10b981; font-family: monospace;",
			"color: #e4e4e7; font-family: monospace; font-size: 12px;",
			"color: #6366f1; font-family: monospace; font-size: 12px; font-weight: bold;",
		);
	}, []);

	return (
		<HelmetProvider>
			<UIProvider>
				<HashRouter>
					<ScrollToTop />
					<BackgroundLayer />
					{/* Removed bg-zinc-950 to allow background components to show through */}
					<div className="flex flex-col min-h-screen text-zinc-100 selection:bg-indigo-500/30 relative">
						<Header />
						<main className="flex-grow container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 max-w-7xl relative z-10">
							<Routes>
								<Route path="/" element={<Home />} />
								<Route path="/idea/:id" element={<IdeaDetail />} />
							</Routes>
						</main>
						<Footer />
						<ContactModal />
					</div>
				</HashRouter>
			</UIProvider>
		</HelmetProvider>
	);
};

export default App;

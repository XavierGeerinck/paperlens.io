import React from "react";
import {
	X,
	Mail,
	Github,
	Twitter,
	Linkedin,
	Send,
	Copy,
	Check,
	Globe,
} from "lucide-react";
import { useUI } from "../context/UIContext";
import { USER_CONFIG } from "../config";

const ContactModal: React.FC = () => {
	const { isContactOpen, closeContact } = useUI();
	const [copied, setCopied] = React.useState(false);

	if (!isContactOpen) return null;

	const handleCopyEmail = () => {
		navigator.clipboard.writeText(USER_CONFIG.email);
		setCopied(true);
		setTimeout(() => setCopied(false), 2000);
	};

	const socialLinks = [
		...(USER_CONFIG.social.linkedin
			? [
					{
						label: "LinkedIn",
						icon: <Linkedin className="w-5 h-5" />,
						url: USER_CONFIG.social.linkedin,
						desc: "Professional Network",
					},
				]
			: []),
		...(USER_CONFIG.social.website
			? [
					{
						label: "Personal Blog",
						icon: <Globe className="w-5 h-5" />,
						url: USER_CONFIG.social.website,
						desc: "Articles & Tutorials",
					},
				]
			: []),
		...(USER_CONFIG.social.github
			? [
					{
						label: "GitHub",
						icon: <Github className="w-5 h-5" />,
						url: USER_CONFIG.social.github,
						desc: "Codebase & PRs",
					},
				]
			: []),
	];

	return (
		<div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
			{/* Backdrop */}
			<div
				className="absolute inset-0 bg-zinc-950/80 backdrop-blur-sm transition-opacity"
				onClick={closeContact}
			/>

			{/* Modal Content */}
			<div className="relative w-full max-w-lg bg-zinc-900 border border-zinc-800 shadow-2xl animate-in fade-in zoom-in-95 duration-200">
				{/* Header */}
				<div className="flex items-center justify-between p-6 border-b border-zinc-800 bg-zinc-950/50">
					<div className="flex items-center gap-3">
						<div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
						<h2 className="text-lg font-mono font-bold text-white tracking-widest uppercase">
							Initialize Connection
						</h2>
					</div>
					<button
						onClick={closeContact}
						className="text-zinc-500 hover:text-white transition-colors"
					>
						<X className="w-5 h-5" />
					</button>
				</div>

				{/* Body */}
				<div className="p-6 space-y-6">
					<p className="text-zinc-400 text-sm leading-relaxed">
						I am always open to collaboration on high-impact moonshot projects.
						Connect with me on LinkedIn or check out my latest writing.
					</p>

					{/* Primary Action: Email */}
					<div className="p-4 bg-zinc-950 border border-zinc-800 flex flex-col gap-3 group hover:border-indigo-500/50 transition-colors">
						<div className="flex items-center justify-between">
							<div className="flex items-center gap-3">
								<div className="p-2 bg-indigo-500/10 rounded">
									<Mail className="w-5 h-5 text-indigo-400" />
								</div>
								<div>
									<div className="text-xs font-mono text-zinc-500 uppercase tracking-wider">
										Direct Message
									</div>
									<div className="text-white font-medium">
										{USER_CONFIG.email}
									</div>
								</div>
							</div>
							<div className="flex gap-2">
								<button
									onClick={handleCopyEmail}
									className="p-2 hover:bg-zinc-800 text-zinc-500 hover:text-white transition-colors rounded"
									title="Copy Email"
								>
									{copied ? (
										<Check className="w-4 h-4 text-green-500" />
									) : (
										<Copy className="w-4 h-4" />
									)}
								</button>
								<a
									href={`mailto:${USER_CONFIG.email}`}
									className="p-2 bg-white text-black hover:bg-zinc-200 transition-colors rounded"
									title="Send Email"
								>
									<Send className="w-4 h-4" />
								</a>
							</div>
						</div>
					</div>

					{/* Secondary Actions: Grid */}
					<div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
						{socialLinks.map((link) => (
							<a
								key={link.label}
								href={link.url}
								target="_blank"
								rel="noreferrer"
								className="flex flex-col items-center justify-center p-4 bg-zinc-900 border border-zinc-800 hover:bg-zinc-800 hover:border-zinc-700 transition-all group text-center"
							>
								<div className="mb-3 text-zinc-400 group-hover:text-white transition-colors">
									{link.icon}
								</div>
								<div className="text-sm font-bold text-zinc-200 mb-1">
									{link.label}
								</div>
								<div className="text-[10px] text-zinc-500 font-mono uppercase">
									{link.desc}
								</div>
							</a>
						))}
					</div>
				</div>

				{/* Footer decoration */}
				<div className="h-1 w-full bg-gradient-to-r from-indigo-500 via-purple-500 to-zinc-800" />
			</div>
		</div>
	);
};

export default ContactModal;

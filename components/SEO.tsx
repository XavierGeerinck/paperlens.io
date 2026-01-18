import React from "react";
import { Helmet } from "react-helmet-async";
import { USER_CONFIG } from "../config";

interface SEOProps {
	title?: string;
	description?: string;
	image?: string;
	type?: "website" | "article";
	url?: string;
	date?: string;
}

const SEO: React.FC<SEOProps> = ({
	title,
	description,
	image,
	type = "website",
	url,
	date,
}) => {
	const siteTitle = USER_CONFIG.name + " | " + USER_CONFIG.lab;
	const pageTitle = title ? `${title} | ${USER_CONFIG.lab}` : siteTitle;
	const metaDescription =
		description || USER_CONFIG.bio.replace("%NAME%", USER_CONFIG.name);
	const metaImage = image || USER_CONFIG.avatar;
	const siteUrl = USER_CONFIG.social.website;
	const currentUrl = url || siteUrl;

	// Structured Data (JSON-LD)
	const schemaData =
		type === "article"
			? {
					"@context": "https://schema.org",
					"@type": "Article",
					headline: title,
					description: metaDescription,
					image: metaImage,
					author: {
						"@type": "Person",
						name: USER_CONFIG.name,
						url: USER_CONFIG.social.website,
					},
					publisher: {
						"@type": "Organization",
						name: USER_CONFIG.lab,
						logo: {
							"@type": "ImageObject",
							url: USER_CONFIG.avatar,
						},
					},
					datePublished: date,
					mainEntityOfPage: {
						"@type": "WebPage",
						"@id": currentUrl,
					},
				}
			: {
					"@context": "https://schema.org",
					"@type": "WebSite",
					name: USER_CONFIG.lab,
					url: siteUrl,
				};

	return (
		<Helmet>
			{/* Basic */}
			<title>{pageTitle}</title>
			<meta name="description" content={metaDescription} />
			<meta name="image" content={metaImage} />

			{/* Canonical URL - critical for SEO */}
			<link rel="canonical" href={currentUrl} />

			{/* Robots */}
			<meta name="robots" content="index, follow" />

			{/* Open Graph */}
			<meta property="og:site_name" content={USER_CONFIG.lab} />
			<meta property="og:title" content={pageTitle} />
			<meta property="og:description" content={metaDescription} />
			<meta property="og:image" content={metaImage} />
			<meta property="og:type" content={type} />
			<meta property="og:url" content={currentUrl} />

			{/* Twitter */}
			<meta name="twitter:card" content="summary_large_image" />
			<meta name="twitter:creator" content="@XavierGeerinck" />
			<meta name="twitter:title" content={pageTitle} />
			<meta name="twitter:description" content={metaDescription} />
			<meta name="twitter:image" content={metaImage} />

			{/* Structured Data */}
			<script type="application/ld+json">{JSON.stringify(schemaData)}</script>
		</Helmet>
	);
};

export default SEO;

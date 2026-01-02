import React from "react";
import { Helmet } from "react-helmet-async";
import { USER_CONFIG } from "../config";

interface SEOProps {
	title?: string;
	description?: string;
	image?: string;
	type?: "website" | "article";
	url?: string;
}

const SEO: React.FC<SEOProps> = ({
	title,
	description,
	image,
	type = "website",
	url,
}) => {
	const siteTitle = USER_CONFIG.name + " | " + USER_CONFIG.lab;
	const pageTitle = title ? `${title} | ${USER_CONFIG.lab}` : siteTitle;
	const metaDescription =
		description || USER_CONFIG.bio.replace("%NAME%", USER_CONFIG.name);
	const metaImage = image || USER_CONFIG.avatar;
	const siteUrl = USER_CONFIG.social.website;
	const currentUrl = url || siteUrl;

	return (
		<Helmet>
			{/* Basic */}
			<title>{pageTitle}</title>
			<meta name="description" content={metaDescription} />
			<meta name="image" content={metaImage} />

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
		</Helmet>
	);
};

export default SEO;

import type { Site, SocialObjects } from "./types";

export const SITE: Site = {
    website: "https://winter.sci.dev/", // replace this with your deployed domain
    author: "KO DONGHYO",
    //   profile: "https://satnaing.dev/",
    desc: "KO DONGHYO, coffee-science geek, application & embeded developer's personal blog",
    title: "❄️ winter.sci.dev",
    ogImage: "wintergreen.png",
    lightAndDarkMode: true,
    postPerIndex: 4,
    postPerPage: 3,
    // scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
    showArchives: true,
    // editPost: {
    //     url: "https://github.com/satnaing/astro-paper/edit/main/src/content/blog",
    //     text: "Suggest Changes",
    //     appendFilePath: true,
    // },
};

export const LOCALE = {
    lang: "ko", // html lang code. Set this empty and default will be "en"
    langTag: ["ko-KR"], // BCP 47 Language Tags. Set this empty [] to use the environment default
} as const;

export const LOGO_IMAGE = {
    enable: false,
    svg: true,
    width: 216,
    height: 46,
};

export const SOCIALS: SocialObjects = [
    {
        name: "Github",
        href: "https://github.com/enzoescipy",
        linkTitle: ` ${SITE.title} on Github`,
        active: true,
    },
    {
        name: "Instagram",
        href: "https://www.instagram.com/winter.sci.dev/",
        linkTitle: `${SITE.title} on Instagram`,
        active: true,
    },
    {
        name: "LinkedIn",
        href: "https://www.linkedin.com/in/donghyo-ko-a53a55344/",
        linkTitle: `${SITE.title} on LinkedIn`,
        active: true,
    },
    {
        name: "Mail",
        href: "mailto:enzoescipy@gmail.com",
        linkTitle: `Send an email to ${SITE.title}`,
        active: false,
    },
    {
        name: "YouTube",
        href: "https://github.com/satnaing/astro-paper",
        linkTitle: `${SITE.title} on YouTube`,
        active: false,
    },
];

export const GISCUS = {
    repo: 'enzoescipy/winter.sci.dev', // Replace with your repo
    repoId: 'R_kgDONnJ-yg', // Get from giscus.app
    category: 'General',
    categoryId: 'DIC_kwDONnJ-ys4CxDO1', // Get from giscus.app
    mapping: 'pathname',
    reactionsEnabled: '1',
    emitMetadata: '0',
    inputPosition: 'bottom',
    theme: 'light',
    lang: 'ko',
} as const;
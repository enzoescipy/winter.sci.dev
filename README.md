# winter.sci.dev - Wintergreen Labs

[Wintergreen-Lab Cheif Winter's Blog](https://www.winter-sci-dev.com/about/)

winter.sci.dev is the digital home of **Wintergreen Labs** (겨울녹 연구소), built as a highly customized fork of [AstroPaper](https://github.com/satnaing/astro-paper), a minimal, responsive, accessible, and SEO-friendly Astro blog theme originally designed and crafted by [Sat Naing](https://satnaing.dev).

We have extensively modified AstroPaper to suit our needs, including integration with Docker for easy deployment, custom components for lab projects, and optimizations for our research in AI, programming patterns, and creative systems like the Weaver framework. This site serves as a public showcase for our prototypes, blog posts on technical explorations, and documentation for ongoing experiments.

Wintergreen Labs focuses on extracting and applying innovative patterns from AI systems (like Weaver) to real-world applications, starting with tools for creativity and productivity. Check out our [projects page](src/pages/projects.astro) or blog for more.

This fork follows best practices from the original while adding our unique enhancements. Light and dark modes are supported, along with fuzzy search, pagination, and more.

## Credits

- **Original Author:** [Sat Naing](https://satnaing.dev) – Thank you for AstroPaper!
- **Fork & Customizations:** Wintergreen Labs (고동효 & his secret partner)
- **License:** MIT (same as original)


## Project Structure

The structure is based on AstroPaper with additions for our lab:

```bash
/
├── public/                 # Static assets, logos, favicon
├── src/
│   ├── assets/             # Images, icons
│   ├── components/         # React components (e.g., Comments, custom UI)
│   ├── content/
│   │   ├── blog/           # Markdown blog posts
│   │   └── config.ts       # Content config
│   ├── layouts/            # Page layouts
│   ├── pages/              # Astro pages (blog, projects, about)
│   ├── styles/             # Tailwind & global styles
│   ├── utils/              # Helpers (e.g., theme utils)
│   ├── config.ts           # Site config
│   └── types.ts            # TypeScript types
├── docker-compose.yml      # Docker setup
├── Dockerfile              # Container build
├── nginx.conf              # Production server config
└── package.json            # Dependencies
```

Blog posts in `src/content/blog/`. New pages in `src/pages/`.

## Documentation

Refer to the original AstroPaper docs for core usage, adapted here:

- [Configuration Guide](https://astro-paper.pages.dev/posts/how-to-configure-astropaper-theme/)
- [Adding Posts](https://astro-paper.pages.dev/posts/adding-new-posts-in-astropaper-theme/)
- [Customizing Colors](https://astro-paper.pages.dev/posts/customizing-astropaper-theme-color-schemes/)

Our custom docs:
- Docker Deployment: See below
- Weaver Integration: In development; check blog for updates

## Tech Stack

**Core:**
- [Astro](https://astro.build/) (Build tool)
- [TypeScript](https://www.typescriptlang.org/)
- [React](https://react.dev/) (Components)
- [TailwindCSS](https://tailwindcss.com/)

**Tools:**
- [Fuse.js](https://fusejs.io/) (Search)
- [Boxicons](https://boxicons.com/) & [Tabler Icons](https://tabler-icons.io/)
- [Prettier](https://prettier.io/) (Formatting)
- [ESLint](https://eslint.org) (Linting)

**Deployment:**
- [Docker](https://docker.com)
- [Nginx](https://nginx.com) (Prod server)
- Let's Encrypt (SSL)

**Other:**
- Giscus (Comments)

## Running Locally

Clone this repo and install:

```bash
npm install
npm run dev  # http://localhost:4321
```

Or use Docker:

```bash
# Build image
docker build -t winter-sci-dev .

# Dev run
docker run -p 4321:80 -e DISABLE_SSL=true winter-sci-dev

# Prod build & run
docker build -t winter-sci-dev .
docker run -p 80:80 -p 443:443 
  -e DOMAIN=winter.sci.dev 
  -e EMAIL=enzoescipy@gmail.com 
  -v /etc/letsencrypt:/etc/letsencrypt 
  -v /var/lib/letsencrypt:/var/lib/letsencrypt 
  winter-sci-dev
```

For docker-compose:
```bash
docker compose up -d
```

## Google Site Verification (Optional)

Add to `.env`:
```bash
PUBLIC_GOOGLE_SITE_VERIFICATION=your-value
```

## Commands

| Command | Action |
|---------|--------|
| `npm run dev` | Local dev server |
| `npm run build` | Build for production |
| `npm run preview` | Preview build |
| `npm run format` | Format code |
| `npm run lint` | Lint code |
| `docker build -t winter-sci-dev .` | Build Docker image |
| `docker run ...` | Run container (see above) |
| `docker compose up` | Compose services |


## Feedback

Open an issue on GitHub or email us. Contributions welcome!

Contact: enzoescipy@gmail.com

## License

MIT License © 2023 Sat Naing (original), extended by Wintergreen Labs winter.sci.dev (enzoescipy)

---

Forked with gratitude from [AstroPaper](https://github.com/satnaing/astro-paper) by [Sat Naing](https://satnaing.dev).
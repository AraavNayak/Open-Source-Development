Q: How are webites deployed?

Physical / On-Premises Servers
This is the traditional approach. You own or rent physical hardware in a data center (or even your own office). You're responsible for everything: the hardware, OS, networking, power, cooling, and security. It gives you full control but requires significant upfront cost and maintenance.

Virtual Machines (Cloud IaaS)
Services like AWS EC2, Google Compute Engine, and Azure VMs let you rent virtualized servers in the cloud. You don't own physical hardware, but you still manage the OS and everything above it. This is very popular because it's flexible and scalable — you can spin up or shut down servers in minutes.

Platform as a Service (PaaS)
Services like Heroku, Render, Railway, and Google App Engine abstract away the server management. You just push your code and the platform handles the OS, runtime, scaling, etc. Much less DevOps overhead.

Containers (Docker / Kubernetes)
You package your app and its dependencies into a container (Docker image), then deploy it anywhere containers run. Kubernetes orchestrates multiple containers at scale. This is very common in modern production systems. Cloud providers offer managed Kubernetes: AWS EKS, Google GKE, Azure AKS.

Serverless / Functions
Services like AWS Lambda, Cloudflare Workers, and Vercel Edge Functions let you deploy individual functions rather than a whole server. The cloud provider handles all infrastructure. You only pay for actual execution time. Great for APIs and event-driven workloads.

Static Site Hosting
For frontend-only sites (HTML/CSS/JS), platforms like Vercel, Netlify, and GitHub Pages deploy your files to a global CDN. There's no server to manage at all — files are just served from edge locations worldwide.

How the code actually gets there
Regardless of the target, the deployment process typically looks like:

Code is pushed to a Git repo (GitHub, GitLab, etc.)
CI/CD pipeline triggers (GitHub Actions, Jenkins, CircleCI) — runs tests, builds the app
Artifact is produced — a compiled binary, Docker image, or zip of files
Artifact is pushed to the target — uploaded to a server via SSH/SCP, pushed to a container registry, or deployed via a platform's CLI/API


In short, the spectrum goes from most control / most work (physical servers) to least control / least work (serverless/static hosting). Most modern apps land somewhere in the middle — usually containers on cloud VMs or a PaaS.
# Prefix
I asked AI to do a Go-Bang game website and I uploaded it to university's personal webpage.
<center><img src=./GO-Bang.png width='300'></center>
And I was also considering building github.io page.
But after helping cdl to online his webpage, I suddenly came to interest in build it on my server. Previously I saw **Cloudflare Tunnel** And I decided to have a dive.

# Github.io page
Using lovable.dev to develop a personal website, with dark mode, and supports both English and Chinese.
Using Chrome Extension **Lovable Project Download** to download the project. The thing is, the downloaded project could not be used directly, it needs configuration or compilation, which takes me one day to figure out.
Using Node.js:
```shell
# Under Project source root:
npm run build

# If vite not installed, do:
npm install --save-dev @vitejs/plugin-react
```

And in `./dist` folder is the completed webpage.
# Cloudflare Tunnel
Domain name required.

Bought a new domain name.
I tried to connect it with my github.io page, and it won't work. I also tried with 
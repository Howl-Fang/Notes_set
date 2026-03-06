My friend: cdl.

## 20260306
Today he asked me for help, loading his webpage online. He started a trial at [aliyun](http://cn.aliyun.com), which gives a cloud server. He also bought a domain name.

As building websites has restrictions in mainland China, censorship is asked to be made on domain names, SSL licenses, and website its self, and the webserver is required to be bound with ID.

I first thought that the server has no public IP address, and some other operations is required. After binding the domain name with the IP address, I tried if I can access ssh with bare domain name, and it surprisingly succeeded. Now I realized that the server does have public IP address.

Aliyun provides a dashboard for quick access, and I installed Nginx from it. I tried accessing locally or remotely, but it can't work. The first issue I met was the ownership of the index.html, after solving it, I added a rule on the firewall. And it finally succeed.

But after minutes, the webpage shows it's not censored, so then I can only wait for cdl to finish the administrative issues.
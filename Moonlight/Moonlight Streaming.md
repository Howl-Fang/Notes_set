<center><font size=5><b><a href="https://moonlight-stream.org">Moonlight Streaming</a> to Mac</b></font></center>
*Recollection Version*

I asked my friend to set up a mini PC for me, to fulfill my need for a PC, as I have a MacBook already. I want a environment that I used to develop things, yet another need is to enjoy video games.

The setup is nearly out-of-the-box. No extra explanation is needed. Yet some small bugs occurs from time to time.

### Headless display
I bought a Video Capture Card, and luckily I chose the one with power supply. So it becomes headless adapter when it's not connected to display device. However it can only work on 1920 x 1080 at most, which is not enough and does not fit the screen ratio of my mac.
I found an app [Parsec Virtual Display](https://github.com/nomi-san/parsec-vdd), which is originally for [Parsec](https://parsec.app), the backup one as it's in poorer quality. Virtual displays could be added into it so it can render for moonlight.
Yet there are some issues, Parsec VD sometimes cannot launch when windows startup, so capture card is still needed. Also, after using capture card, it has to be re-plugged to PC or it will not work.

### AWDL caused jam
Details in ./AWDL submodule.


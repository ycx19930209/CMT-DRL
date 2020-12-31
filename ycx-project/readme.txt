放置路径：/P4/tutorials/exercises$ 
sudo make
xterm h3(service-get1)，h5(service_get2), h4(client_get), s1(w-wetter,learning，new-setter)，
s2(tcpdump -i s2-eth1 -w s2-eth1.pcap)

h1: iperf  -s -p 5566 -i 1                                 10.0.1.1
h2:  iperf -c 10.0.1.1 -p 5566 -i 1 -t 1000                10.0.2.2


xterm s1    simple_switch_CLI ,看看能不能执行
## Interlude: review of logic

### Exercise 1: Compute the truth table for NOT:
### Answer:

| X | NOT(X)|
|:-:|:-----:|
|0  |1      |
|1  |0      |

### Exercise 2: Compute the truth table table for AND.
### Answer:

|X   | Y | AND(X,Y)|
|:--:|:-:|:-------:|
|0   |0  |0        |
|1   |0  |0        |
|0   |1  |0        |
|1   |1  |1        |

### Exercise 3: Compute the truth table for exclusive-or, defined by the formula:

XOR(X, Y) = (X OR Y) AND NOT (X AND Y)
### Answer:

|X  |Y  |XOR(X, Y)|
|:-:|:-:|:-------:|
|0  |0  |0        |
|1  |0  |1        |
|0  |1  |1        |
|1  |1  |0        |

### Exercise 4: Prove De Morgan's theorem, NOT(X OR Y) = NOT(X) AND NOT(Y),
by completing the table and checking the last two columns are the same.
### Answer:

| X | Y | NOT(X OR Y) | NOT(X) AND NOT(Y)|
|:-:|:-:|:-----------:|:----------------:|
| 0 | 0 | 1           | 1                |
| 1 | 0 | 0           | 0                |
| 0 | 1 | 0           | 0                |
| 1 | 1 | 0           | 0                |


### Exercise 5: using truth tables, check these three equations
### Answer:

|X   | Y   |NOT(X)|NAND(1,X)|AND(X,Y)|NOT(NAND(X,Y))|OR(X,Y)|NAND(NOT(X),NOT(Y)))|
|:--:|:---:|:----:|:-------:|:------:|:------------:|:-----:|:------------------:|
|0   |0    |1     |1        |0       |0             |0      |0                   |
|1   |0    |0     |0        |0       |0             |1      |1                   |
|0   |1    |1     |1        |0       |0             |1      |1                   |
|1   |1    |0     |0        |1       |1             |1      |1                   |

### Exercise 6: write similar formulas expressing NOT, AND, and OR in terms of NOR
### Answer:

* NOT(X) = NOR(0,X)
* AND(X,Y) = NOR(NOT(X),NOT(Y)))
* OR(X,Y) = NOT(NOR(X,Y))

|X | Y | NOT(X) | NOR(0,X) | AND(X,Y) | NOR(NOT(X),NOT(Y))) | OR(X,Y) | NOT(NOR(X,Y)) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|0   |0  |1  |1  |0  |0  |0  |0  |
|0   |1  |1  |1  |0  |0  |1  |1  |
|1   |0  |0  |0  |0  |0  |1  |1  |
|1   |1  |0  |0  |1  |1  |1  |1  |

### Exercise 7: why NOT and OR can't be expressed in terms of AND? Explain
### Answer:

Because it breaks De Morgan's laws.


## Binary numbers as lists of boolean values

### Exercise 8: Without listing explicitly, how many possible 8-bit binary numbers are there?
### Answer:

There are 2^8 (256) different possible values for 8 bits. When interpreted as an unsigned integer, the possible values range from 0 to 255; when signed, the values are −128 to 127.



### Exercise 9: Convert X = 110 to decimal.
### Answer:


  1 * 2^2 + 1 * 2^1 + 0 * 2^0
= 1 * 4   + 1 * 2   + 0 * 1
=   4     +   2     +   0
= 6

### Exercise 10: Convert 11 to binary.
### Answer:

2^0 = 1 <= 11
2^1 = 2 <= 11
2^2 = 4 <= 11
2^3 = 8 <= 11
2^4 = 16 > 11 => high bit is 3

Thus X = X3 X2 X1 X0, and X3 = 1. That’s how we get a binary 1000.

11 - 8 = 3, so now we find the highest bit of 3:

2^0 = 1 <= 3
2^1 = 2 <= 3
2^2 = 4 > 3 => high bit is 1

That’s how we get a binary 10.

3 - 2 = 1, so now we find the highest bit of 1:

2^0 = 1 => high bit is 0

That’s how we get a binary 1.

(in binary) 1000 + 10 + 1 = 1011.


### Exercise 11: Convert these powers of 2 into binary: 2, 4, 8, 16, 32. What do you notice?
### Answer:


* 2 = 10, 4 = 100, 8 = 1000, 16 = 10000, 32 = 100000
* I notice that every new power of 2 requires adding a new bit.

### Exercise 12: Convert these numbers into binary: 1, 3, 7, 15, 31 (they are all 2^n - 1 for some n). What do you notice?
### Answer:


* 1 = 1, 3 = 11, 7 = 111, 15 = 1111, 31 = 11111
* As these decimal numbers turn to powers of two after adding 1, we need one more digit to write them in binary. E.g. if 1 in binary is 1 and all we have are zeroes and ones, we need to add one more digit to express 2 and so we get 10.

### Exercise 13: check that these numbers all have the same 3-bit representation
### Answer:

* For the case 3 = 11 = 17: (011) i.e. 3 equals to 1(011) i.e. 11 but doesn’t equal to 10(001) i.e. 17. There is a mistake. We need a 19 instead of a 17.
* For the 0 = 8 = 16: (000) i.e. 0 equals to 1(000) i.e. 8 and equals to 10(000) i.e. 16. Everything is correct.
* For the case 2 = 10 = 18: (010) i.e. 2 equals to 1(010) i.e. 10 and equals to 10(010) i.e. 18. Everything is correct, too.


## Binary arithmetic as logical operations

### Exercise 14: complete the table by converting 2 into single-bit binary:
### Answer:

|X0 |Y0 |Z0 |
|:-:|:-:|:-:|
|0  |0  |0  |
|1  |0  |1  |
|0  |1  |1  |
|1  |1  |0  |


### Exercise 15: do the same for single-bit multiplication: write down the table of binary numbers for X0, Y0, and the binary representation of their product Z0, and find the logical operation which matches.

### Answer:

|X0 |Y0 |Z0 |
|:-:|:-:|:-:|
|0  |0  |0  |
|1  |0  |0  |
|0  |1  |0  |
|1  |1  |1  |

The logical operation which matches the output Z0 is AND.

## Digital Logic

### Exercise 16: Using A and B as the inputs, and OUT as the output, explain how this circuit acts as NAND(A,B); for each entry in the truth table, follow the explanation above. True is "high energy" and False is "low energy".
### Answer:


|A  |B  |NAND(A,B)|
|:-:|:-:|:-------:|
|0  |0  | 1       |
|1  |0  | 1       |
|0  |1  | 1       |
|1  |1  | 0       |

The output of the NAND Gate is LOW if and only if both of its inputs are HIGH otherwise its output is HIGH, note that the output of the NAND Gate is exact complement of the AND Gate in which output will be HIGH if only if inputs are HIGH otherwise the output is LOW for all other combinations of the input variables.


## Name resolution and Routing

### Exercise 17: show that every IPv4 can be represented by four 8bit unsigned integers, and that every 8bit unsigned integer is between 0 and 255.
### Answer:

The IPv4 address is a 32-bit number that uniquely identifies a network interface on a machine. An IPv4 address is typically written in decimal digits, formatted as four 8-bit fields separated by periods. Each 8-bit field represents a byte of the IPv4 address. This form of representing the bytes of an IPv4 address is often referred to as the dotted-decimal format. Dotted decimal notation can be converted to its binary equivalent by using weighted binary bits notated with 2^n where n is the number of bits.


### Exercise 18: how many IPv4 addresses are there? Is it enough? Explain.
### Answer:

There 4294967296 addresses in IPv4 (it equals to 2^32). The shortage of IPv4 addresses is a big problem for the development of the Internet and digital infrastructure. It is impossible to fully deploy the Internet of Things in IPv4, even Internet providers meet this problem, which greatly increases the cost of the networks and complicates their architecture. For users, this situation also promises little fun. Many sites which previously let users access their pages without any problems will eventually have to require filling in captcha to confirm that the visitor is not a robot.

### Exercise 19: use ping in a terminal to resolve a domain name. Copy-paste the command you used, and the result.
### Answer:


(base) MacBook-Air-Ila:~ datanikitin$ ping -c 10 vk.com
PING vk.com (93.186.225.208): 56 data bytes
64 bytes from 93.186.225.208: icmp_seq=0 ttl=55 time=53.735 ms
64 bytes from 93.186.225.208: icmp_seq=1 ttl=55 time=51.208 ms
64 bytes from 93.186.225.208: icmp_seq=2 ttl=55 time=50.768 ms
64 bytes from 93.186.225.208: icmp_seq=3 ttl=55 time=50.760 ms
64 bytes from 93.186.225.208: icmp_seq=4 ttl=55 time=52.984 ms
64 bytes from 93.186.225.208: icmp_seq=5 ttl=55 time=50.348 ms
64 bytes from 93.186.225.208: icmp_seq=6 ttl=55 time=53.765 ms
64 bytes from 93.186.225.208: icmp_seq=7 ttl=55 time=51.594 ms
64 bytes from 93.186.225.208: icmp_seq=8 ttl=55 time=52.239 ms
64 bytes from 93.186.225.208: icmp_seq=9 ttl=55 time=52.091 ms

--- vk.com ping statistics ---
10 packets transmitted, 10 packets received, 0.0% packet loss
round-trip min/avg/max/stddev = 50.348/51.949/53.765/1.171 ms 


## Bandwidth, latency, reliability

### Exercise 20: The Multipath TCP project aims to allow TCP packets to be split across multiple network links and reassembled at the destination. For example, if you were uploading a 100 megabyte file to a server from your phone, it would allow you to send 75 megabytes by WiFi and 25 megabytes by cellular automatically. How should the ratio be chosen if you want to minimise transmission time? Minimise cellular bandwidth use? Explain.
### Answer:


If I want to minimise transmission time and cellular bandwidth and WiFi bandwidth are equal, I should choose a 50%/50% ratio.
If I want to minimise cellular bandwidth use, I should send the whole file by WiFi.

### Exercise 21: UDP is popular for streaming media; explain why.
### Answer:

By its nature, UDP is not reliable — messages may be lost or delivered out of order. By adding loss detection and retransmission mechanisms, reliable multicast has been implemented on top of UDP or IP by various middleware products, e.g. those that implement the Real-Time Publish-Subscribe (RTPS) Protocol of the Object Management Group (OMG) Data Distribution Service (DDS) standard, as well as by special transport protocols such as Pragmatic General Multicast (PGM).

### Exercise 22: Read the Wikipedia articles on multicast and anycast routing. Why is anycast good for content delivery networks, and why is multicast good for live-streaming? What are some other uses for these?
### Answer:

*Anycast is good for content delivery networks because most HTTP connections to such networks request static content such as images and style sheets, they are generally short-lived and stateless across subsequent TCP sessions. The general stability of routes and statelessness of connections makes anycast suitable for this application, even though it uses TCP.

*Multicast is good for live-streaming because IP multicast scales to a larger receiver population by not requiring prior knowledge of who or how many receivers there are. Multicast uses network infrastructure efficiently by requiring the source to send a packet only once, even if it needs to be delivered to a large number of receivers. The nodes in the network take care of replicating the packet to reach multiple receivers only when necessary.

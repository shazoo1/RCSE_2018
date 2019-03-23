# Ideas for course works

## 1. Datacenter client profiling
### Prerequisites
You have a data set with the following data:

1. Virtual machine characteristics
    - CPU
    - Memory
    - HDD
    
2. Virtual machine workload
    - CPU Usage percentage
    - CPU Ready time
    - Memory consumption
    - HDD waiting time

3. Hypervisors workload
    - CPU Usage percentage
    - Memory consumption

4. Storage workload
    - HDD waiting time
    - IO per seconds
    
5. Network traffic
    - netflow from public network
    
### Task
Your main aim is to help the datacenter to find information about typical workload in their environment. To be more conscious, it may be helpful to answer on the following questions:

1. What is average CPU workload in the entire cluster?
2. What is average resource consumption for a virtual machine?
3. How network, CPU and Memory consumption  at average change during a working day?
4. Which typical client's presets exist? How many VM's cluster groups can be found?
5. What is a typical workload for each client profile?


There are might be found be hardcore, yet interesting questions to be answers:

1. How many errors occurs each days? What category are they?
2. Is amount of errors occuring correlates with the virtualization system  workload

And the most difficult questions are:

1. What application can be run on each of client's configuration? e.g. the analysis should be able to answer on the question if is it possible to run Windows SQL Server on CPU 2, Memory 2, HDD 50, and so on.
2. Build a recommend system, where a client choose which application he'd to run, and the system should be able to suggest the configuration profile.
    
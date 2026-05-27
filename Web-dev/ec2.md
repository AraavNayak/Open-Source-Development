

Companies are increasingly migrated workloads from physical servers to cloud. Ehances scalability adn easily adjust resources to changing demands. Cost effective; eliminates need for upfront infrastructure/hardwre payments.

On premises infra requires upfront and ongoing capex investments + expertise in configuration setup management and scaling.

Cloud avoids this issues.

Think of AWS like a massive physical computer in a data center. An instance is just a slice of that computer rented to you — your own virtual machine with some CPU, RAM, and storage.

Amazon EC2 (Elastic Compute Cloud) provides secure/resizable capacity for any workload. Allows you to run anything, eg. basic web apps or complex ML models. No upfront costs for migrating to EC2, Pay as you go model.

EC2 offers broadest and deepest selection of instances...allowing customers to select optimal balance of comput, memory, storage, and network resources to tailor infra to workload needs.
EC2 instances are scalable and cusotmizable to changing demand. EC2 seamlessly integrates with other AWS services ()

To use:
Create an AWS account and then launch EC2 instance.

Sources:
https://www.youtube.com/watch?v=t48aVpw6kkI&t=22s


----------------------------------------

Related concepts:
- Auto scaling
- Load balancing
- instances are powered by specialized and general-purpose chips (eg. AWS graviton processor introduced in 2018)
- AWS Nitro offloads virtualization functions to dedicated hardware and software....delivers all resources of a server to instances.
- AWS Managemnt console




Load Balancing
Imagine a busy restaurant with multiple cashiers. A load balancer is the host at the front door who directs each arriving customer to a free cashier, so no single cashier gets overwhelmed while others sit idle.
In web terms: when millions of users hit your website, a load balancer sits in front of your servers and distributes incoming requests across multiple instances so no single one crashes under pressure.

Auto Scaling
Connected to load balancing — if the restaurant suddenly gets a lunch rush, you call in more cashiers. Auto scaling automatically spins up more instances when traffic spikes, and shuts them down when traffic drops. You only pay for what you use.

AWS Nitro
This one requires understanding a virtualization problem first.
The old way: To run many virtual machines on one physical server, the hypervisor (virtualization software) had to run on the same CPU, stealing ~20–30% of the server's resources just for itself.
Nitro's solution: AWS built dedicated hardware chips that handle virtualization tasks offloaded from the main CPU. The CPU no longer wastes cycles managing virtualization — it's handled by a separate, purpose-built card.
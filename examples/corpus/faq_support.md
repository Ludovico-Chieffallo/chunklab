# Nimbus Cloud Platform — Support FAQ

Frequently asked questions about the Nimbus cloud platform, maintained by the support team and updated with every platform release. Entries are grouped by area; each answer reflects the currently deployed behavior of the platform.

## Accounts and Billing

**How do I change the email address on my account?**
Open Settings, then Profile, and select Edit email. A confirmation link is sent to the new address and the change takes effect only after the link is clicked; until then the platform keeps using the old address for every notification. The previous address receives a security alert with a one-click revert link that stays valid for seventy-two hours, which protects the account if the mailbox was compromised.

**What payment methods are accepted?**
Nimbus accepts the major credit card networks, SEPA direct debit for euro-denominated accounts, and, for annual contracts above 10,000 euros, bank transfer with net-30 payment terms. PayPal and cryptocurrency are not supported in any region. A payment method must be verified with a temporary one-euro authorization before it can be set as the default for a paying organization.

**When am I billed and what does the invoice contain?**
Usage is metered hourly, aggregated per project, and invoiced on the first business day of each month for the previous month. The invoice lists every service line with quantity, unit price, and the project label, so cost allocation across teams needs no external tooling. Invoices are payable within fourteen days of issue, and late accounts receive two reminders before any enforcement begins.

**Can I get a refund for unused prepaid credit?**
Prepaid credit is refundable within sixty days of purchase, minus any amount already consumed, refunded to the original payment method. After sixty days credit becomes non-refundable but never expires, and it is always consumed before the pay-as-you-go balance. Promotional credit granted by the sales team is never refundable and is always consumed first, before any purchased credit is touched.

**Why was my card declined even though it works elsewhere?**
The most frequent causes are an expired card, a bank rule blocking recurring online payments, or a billing-address mismatch with the issuing bank's records. After three consecutive failed charges the account enters a seven-day grace period; during grace the console shows a red banner but nothing is suspended. If the seventh day passes without a successful charge, new resource creation is blocked first, and running compute is stopped forty-eight hours later.

**How do I set a hard spending limit?**
In Billing, open Budgets and create a monthly budget with the hard-cap option enabled. When consumption reaches the cap, new resource creation is rejected immediately and running compute instances are paused, while stored data, snapshots, and reserved addresses are preserved untouched. The cap resets at midnight UTC on the first day of the month, and a webhook can notify your own systems at fifty, eighty, and one hundred percent of the budget.

**Can I move resources between projects?**
Ownership transfer moves a resource and its metrics history to another project in the same organization without downtime. Cost attribution switches at the top of the next hour, and both project owners must approve transfers of resources holding reserved capacity.

**How can my accounting system receive invoices automatically?**
Finalized invoices can be delivered to an SFTP drop or fetched through the invoice API in PDF and structured XML formats. Delivery happens within one hour of invoice finalization, and a checksum manifest accompanies every batch for reconciliation.

## Identity and Access

**What is the difference between members and service accounts?**
Members are human identities that sign in interactively and can hold organization-wide roles, while service accounts are non-interactive identities meant for automation and scoped to a single project by design. Service accounts authenticate with rotating keys rather than passwords, never receive email, and cannot open support tickets. A project can hold at most fifty service accounts, a limit that can be raised through a support request with a justification.

**How does single sign-on work?**
SAML 2.0 and OpenID Connect identity providers can be connected at the organization level, and just-in-time provisioning maps IdP groups to Nimbus roles on first login. When SSO is enforced, password sign-in is disabled for every member except two designated break-glass administrators, whose use triggers an alert to all owners. SCIM synchronization keeps memberships aligned, deprovisioning a member within five minutes of removal in the IdP.

**Can I restrict which regions my teams may use?**
Data residency policies pin a project's resources to an allowed region set, and any attempt to create a resource elsewhere fails with an explicit policy-violation error naming the policy. The restriction covers replicas and backups too, so a compliant project cannot leak data into a disallowed geography even through disaster-recovery features. Policies are versioned, and every change is recorded in the audit trail with the actor and the previous value.

**How do temporary elevated permissions work?**
Just-in-time access lets a member request a higher role for a bounded window between fifteen minutes and eight hours, with a mandatory reason field. Approval comes from any organization owner other than the requester, and the elevation expires automatically without any action from either side. All actions performed during the window are flagged in the audit trail with the elevation id, which makes after-the-fact review a single query.

**What happens to resources when a member leaves?**
Resources always belong to projects, never to members, so removing a member changes ownership of nothing and breaks no running workload. Personal API keys of the removed member are revoked within five minutes, and scheduled jobs they created keep running under the project's service account. A leaving report lists everything the member touched in the last ninety days so the team can review credentials that might need rotation.

**Is multi-factor authentication mandatory?**
Organization owners can require MFA for all members, and new organizations created after January 2026 have the requirement enabled by default. TOTP authenticator apps and FIDO2 hardware keys are supported, while SMS is deliberately not offered as a second factor because of SIM-swap risk. A member without MFA in an enforcing organization is limited to a read-only console session until enrollment is completed.

**Are stale API keys detected?**
Service-account keys older than ninety days are flagged on the credentials hygiene page, and an organization policy can force-expire them. Each key shows its last-used timestamp and source IP, which makes abandoned keys easy to distinguish from quiet-but-alive ones.

**How long do console sessions last?**
Interactive console sessions expire after twelve hours, or after thirty minutes of inactivity when the idle-timeout policy is enabled. Re-authentication always requires the second factor in MFA-enforcing organizations, and active sessions can be revoked centrally per member.

## Virtual Machines

**Which operating system images are maintained by Nimbus?**
Nimbus maintains hardened images for Ubuntu LTS, Debian stable, Rocky Linux, and Windows Server, refreshed within seven days of upstream security releases. Custom images can be imported in qcow2 or VHD format up to 500 GB, and imported images are automatically scanned for known-vulnerable packages. An image catalog per organization lets platform teams pin the exact image versions application teams may launch.

**How do I resize a virtual machine and what is the downtime?**
Stop the instance, choose Resize from the actions menu, select the new flavor, and start the instance again; the boot disk and all attached volumes are preserved exactly as they were. The operation typically completes in about two minutes of downtime, dominated by the reboot itself. Downsizing checks memory pressure first and refuses the operation if the running workload could not have fit in the target flavor during the past hour.

**What exactly happens when I stop an instance?**
A stopped instance releases its CPU and memory reservation but keeps its disks, its private IP, and its device attachments, and you are billed only for storage while it stays stopped. The public IPv4 address is released back to the pool unless it was explicitly reserved as a floating IP beforehand. Instances stopped for more than ninety consecutive days are flagged on the cost-recommendations page as candidates for snapshot-and-delete.

**How do snapshots behave with running workloads?**
Snapshots are incremental and crash-consistent: the first snapshot copies the full disk and later ones store only the blocks that changed since the previous snapshot. For application-consistent captures, the guest agent can quiesce supported databases for up to thirty seconds while the snapshot point is taken. Restoring always creates a new volume and never overwrites the original, so a botched restore cannot destroy the source data.

**Why is my instance in error state and what can I do?**
Error state usually means the host failed capacity checks during a live migration, leaving the instance neither fully on the old host nor started on the new one. Detach and re-attach the floating IP, then use the Rebuild action, which re-creates the instance on healthy hardware while keeping every disk; support can recover the disk in all cases, including a failed rebuild. Error-state minutes are excluded from billing automatically, with no ticket required.

**What is the availability SLA for virtual machines?**
A single instance carries a 99.5 percent monthly availability SLA, while an instance group spread across two availability zones carries 99.95 percent. SLA credits are ten percent of the monthly fee for the affected resources per breached tier, requested through a ticket within thirty days of the incident with the affected instance ids. Maintenance windows announced at least five days ahead are excluded from the SLA computation.

**Does live migration affect my running workload?**
Host maintenance uses live migration, which pauses the guest for under one second while dirty memory pages converge. Latency-sensitive workloads can subscribe to migration events and receive a sixty-second pre-notification to shed traffic before the pause.

**How do I access the serial console of an unreachable instance?**
The serial console is reachable through an SSH tunnel using a short-lived token valid for five minutes, issued per instance and per member. Every serial session is recorded in the audit trail, and a break-glass boot parameter can be injected at next start for password recovery.

## Block Storage and Snapshots

**How many volumes can I attach to one instance?**
Up to sixteen block volumes can be attached to a single instance, hot-plugged while the instance is running. A freshly attached volume appears as a raw device: a filesystem must be created before first use, and the guest needs a rescan only on kernels older than 5.10. The boot volume counts toward the sixteen, so data-heavy designs effectively have fifteen slots for data volumes.

**What performance tiers exist for block storage?**
The standard tier delivers three IOPS per provisioned GB with a floor of one hundred IOPS, the performance tier delivers a flat 7,500 IOPS regardless of size, and the ultra tier provisions IOPS independently from capacity up to 64,000 per volume. Tier changes happen online with no detach required and take effect within ten minutes. Throughput caps scale with instance flavor, so a small instance can bottleneck an ultra volume.

**Can a volume be attached to multiple instances at once?**
Multi-attach is supported only on the ultra tier and must be enabled at volume creation; it cannot be toggled later. The platform provides no locking: a cluster-aware filesystem or an application-level protocol such as a database's shared-disk mode is the customer's responsibility. Snapshots of a multi-attached volume are crash-consistent from the perspective of a single arbitrary attachment, which is rarely what a clustered application wants.

**How are volume snapshots billed?**
Snapshot storage is billed per GB-month of unique blocks after compression, which in practice runs thirty to sixty percent below the logical size for typical workloads. Deleting a snapshot in the middle of a chain never breaks later snapshots, because unique blocks are reference-counted and merged forward. Cross-region snapshot copies are billed as new unique storage in the destination region plus one-time inter-region transfer.

**Can I shrink a volume?**
Volumes can only grow online; shrinking is not supported because safely relocating filesystem blocks below the cut line cannot be guaranteed for arbitrary filesystems. The documented path is to create a smaller volume, attach it alongside, copy data at the filesystem level, and swap mount points during a maintenance window. The cost-recommendations page flags volumes whose used space has stayed under forty percent for thirty days.

**What is volume encryption based on?**
Every volume is encrypted at rest with AES-256-XTS using platform-managed keys by default, with no performance penalty visible to the guest. Projects that must control key custody can enforce customer-managed keys from the key management service through a project policy, and key revocation renders all dependent volumes unreadable within one hour. Encryption in transit between host and storage backend uses mutual TLS on a dedicated storage network.

**Do standard-tier volumes support bursting?**
Standard volumes accumulate burst credits while usage stays below their baseline and can then sustain 3,000 IOPS for up to thirty minutes. The remaining burst balance is exported as a metric, so exhaustion is predictable rather than a surprise slowdown.

**How do I get alerted about a slow volume?**
A volume health event is raised when average latency stays above twenty milliseconds for five consecutive minutes. Health events appear on the volume page and can page on-call through the alerting service like any metric-based rule.

## Object Storage

**What are the object size limits?**
A single PUT accepts objects up to 5 GB, and larger objects up to 5 TB must use multipart upload with parts between 5 MB and 5 GB. Multipart uploads that are started but never completed are garbage-collected after seven days by default, a window that a bucket lifecycle rule can shorten to one day. The part list of an in-progress upload can be inspected, which makes resumable transfer tooling straightforward.

**How durable and available is object storage?**
Objects are erasure-coded across three datacenters within the region, giving eleven nines of designed durability and a 99.9 percent availability SLA on the standard class. Bucket versioning, once enabled, preserves every overwritten or deleted version until a lifecycle rule expires it, which turns accidental deletion into a metadata operation that is fully reversible. Read-after-write consistency is strong for all operations, including overwrites and deletes.

**What storage classes are available and when do they pay off?**
Standard serves hot data with no retrieval fee; Infrequent Access costs roughly forty percent less to store but adds a per-GB retrieval fee and a thirty-day minimum billing duration; Archive is the cheapest, with a twelve-hour restore time and a ninety-day minimum. Lifecycle rules can transition objects automatically by age or by tag, and the transition itself is free. As a rule of thumb, data read less than once a month belongs in Infrequent Access, and data read less than once a year belongs in Archive.

**How do presigned URLs behave?**
A presigned URL embeds a signature that grants time-limited access to exactly one object and one operation, with a maximum validity of seven days. The signature binds the HTTP method and optionally the content hash, so a URL presigned for download cannot be replayed as an upload. The signing machine's clock must be within fifteen minutes of UTC, which is the most common cause of mysterious 403 responses from otherwise correct code.

**What request rates can a bucket sustain?**
Buckets sustain 3,500 write and 5,500 read requests per second per key prefix, and throughput scales linearly by spreading keys across prefixes. Sequential, timestamp-leading key names funnel all traffic into one prefix and are the most common cause of throttling in ingestion pipelines. The platform returns a explicit slow-down error code with a suggested backoff, and the metrics page breaks down request counts per prefix to make hot spots visible.

**How do I safely make part of a bucket public?**
Attach a bucket policy allowing anonymous read on the chosen prefix and disable the public-access block for that bucket, an action that requires an owner-level role. The console then shows a persistent orange badge on the bucket, and the weekly security digest lists every public prefix in the organization. Signed access logs record each anonymous request with source IP and user agent, retained for ninety days by default.

**Is write-once-read-many retention available?**
Object lock in compliance mode prevents deletion or overwrite of a version by any identity until its retention date passes. Governance mode allows privileged bypass with a logged justification, which suits internal policy rather than regulatory retention.

**Can I get a report of everything stored in a bucket?**
Daily inventory reports list every object with size, storage class, encryption status, and checksum, delivered to a destination bucket as compressed CSV or Parquet. Inventories are the recommended input for large-scale audits instead of paginated listing calls.

## Networking

**How do security groups evaluate traffic?**
A security group is a stateful virtual firewall applied at the network interface: rules only allow traffic, there are no deny rules, and return packets of an allowed flow are permitted automatically. Rules can reference other security groups instead of IP ranges, which keeps policies valid as instances come and go. The most frequent misconfiguration is a group that opens ports to external ranges but not to its own group id, which silently blocks traffic between instances in the same subnet.

**Can I peer two private networks, and what are the limits?**
Network peering connects two VPCs in the same region with non-overlapping address ranges, and traffic over the peering stays on the private backbone. Peering is not transitive: a hub-and-spoke of three networks requires three peering connections for full mesh, or a transit gateway once the topology grows past roughly five networks. Route propagation across a peering is explicit, so each side chooses which subnets become reachable.

**Does the platform support IPv6 end to end?**
Dual-stack networking is available in all regions: each subnet receives a /64, addresses are globally routable, and inbound IPv6 traffic is denied by default until a security-group rule opens it. Load balancers, DNS, and the object storage endpoints are dual-stack, while the managed database service remains IPv4-only inside the private network. NAT64 is provided for IPv6-only subnets that must reach legacy IPv4 destinations.

**What network throughput can a single instance reach?**
General-purpose flavors provide up to 4 Gbps of throughput, compute-optimized flavors reach 12.5 Gbps, and the metal series exposes the full 2x25 Gbps of the host NICs. Traffic between availability zones is billed at one cent per GB, while traffic within one zone is free, a difference that matters for chatty replication protocols. Per-flow throughput is capped at 5 Gbps everywhere, so single-stream benchmarks undersell the aggregate.

**How do I set up a site-to-site VPN with automatic failover?**
Create a VPN gateway, define the peer's public IP and shared secret, and configure matching IKEv2 proposals on both ends; two tunnels on distinct gateway hosts are provisioned automatically. Failover between the tunnels is automatic and typically completes within ten seconds, driven by dead-peer detection at five-second intervals. Throughput is capped at 1.5 Gbps per tunnel, and BGP over the tunnels is available for dynamic routing.

**Can I bring my own public IP range?**
Bring-your-own-IP is supported for provider-independent IPv4 blocks of /24 or larger, with a signed letter of authorization from the range holder. Announcement propagation takes up to two business days, during which the range stays usable at its previous location. Brought ranges can back floating IPs and load balancers but not the platform's own service endpoints, and reverse DNS delegation is set up as part of onboarding.

**Are network flow logs available?**
Flow logs record accepted and rejected flows per network interface with one-minute aggregation and land in the logging service or a bucket. Sampling is full by default, and a per-subnet setting can reduce it to one in ten for very chatty fabrics.

**What MTU does the private network support?**
The private network supports jumbo frames up to 8950 bytes, negotiated automatically by the maintained images. Traffic leaving the region is clamped to 1500, and the platform rewrites MSS on egress paths so misconfigured guests avoid silent fragmentation.

## Load Balancers and DNS

**What load balancer types exist?**
The network load balancer operates at layer 4, preserves client source addresses, and handles millions of connections with static per-zone anycast IPs; the application load balancer operates at layer 7 with host- and path-based routing, header rewrites, and native WebSocket support. Both types are billed per hour plus per GB processed. A single project can front the same backends with both types simultaneously, which eases protocol migrations.

**How do health checks decide backend status?**
Each backend is probed on a configurable path and port every ten seconds; three consecutive failures remove it, and two consecutive successes readmit it. Removal only stops new connections: established connections drain for a configurable period up to one hour, which lets long-lived sessions finish cleanly. Health-check results are exported as metrics per backend, so flapping is visible before users notice.

**Can TLS certificates be managed automatically?**
The certificate manager provisions and renews certificates through the integrated ACME client, validating either by DNS delegation or by HTTP token on the balancer itself. Renewal is attempted thirty days before expiry, with alerts at fourteen and seven days if it keeps failing. Custom uploaded certificates are supported for organizations with their own CA, and the balancer serves the certificate chain exactly as uploaded.

**How does DNS hosting handle failover?**
Hosted zones support health-checked records: a primary record answers while its target passes checks, and a secondary record takes over within thirty seconds of failure. Record propagation inside the platform's resolvers is near-instant, while public propagation depends on the record TTL, with sixty seconds as the practical minimum. Zone changes are versioned, and any previous version can be restored with one action.

**Is there protection against volumetric attacks?**
Volumetric DDoS up to 2 Tbps is absorbed by the backbone scrubbing centers at no extra charge, with mitigation engaging automatically within seconds of detection. Application-layer protection with custom rules, rate limits, and bot scoring is part of the web application firewall add-on, billed per million inspected requests. Attack reports with peak rates, vectors, and mitigation timelines appear in the security console within one hour of the event's end.

**Can I use my own domain apex with the load balancer?**
Zone-apex domains cannot point at a CNAME, so the DNS service provides alias records that resolve directly to the balancer's current addresses at no query cost. Alias records track address changes automatically, including during balancer replacement, so the apex never goes stale. For zones hosted elsewhere, a static anycast IP pair can be reserved per balancer as a fallback, with the trade-off that it pins the balancer type.

**Does the application balancer support sticky sessions?**
Cookie-based affinity pins a client to one backend for a configurable duration up to seven days, surviving backend scale-out but not backend removal. The affinity cookie is signed, so clients cannot forge assignments to target a specific backend.

**What is the idle connection timeout?**
The idle timeout defaults to sixty seconds and is extendable to one hour per listener. Long-polling APIs should either raise the timeout or send keep-alive frames, and the balancer emits a distinct metric for connections closed by idle timeout to make tuning evidence-based.

## Kubernetes Service

**Which Kubernetes versions are supported and for how long?**
The managed service tracks the three most recent upstream minor releases, with new minors available within thirty days of upstream. Each minor is supported for approximately fourteen months, and clusters on an end-of-support version are force-upgraded after a sixty-day notice period with three warnings. The API server always runs at most one minor ahead of the oldest supported node pool, matching upstream skew policy.

**How do cluster upgrades avoid downtime?**
Control-plane upgrades are fully managed and zero-downtime, running behind a highly available endpoint. Node pools upgrade by surge: a new node with the target version joins, workloads drain respecting pod disruption budgets, then the old node is removed, repeating pool-wide. The maximum surge and the drain timeout are configurable per pool, and a stuck drain pauses the upgrade rather than force-deleting pods.

**Can node pools scale to zero and how fast do they come back?**
Any node pool except the default system pool can scale to zero. Scale-up from zero takes about ninety seconds, dominated by node provisioning before pods can schedule, and pending pods are the trigger. For latency-sensitive but bursty workloads, the recommended pattern is a minimum of one small node plus aggressive scale-up, which converts the ninety seconds into roughly fifteen.

**Where do container logs go and how long are they kept?**
Stdout and stderr of every container ship automatically to the logging service with a default retention of thirty days, extendable to 365 days per namespace through a logging policy. Log-based metrics and alerts can be defined on any structured field without re-shipping the data. A per-namespace export rule can copy logs to a bucket in near-real time for archival beyond the service's maximum retention.

**How do persistent volumes interact with zones?**
PersistentVolumeClaims are fulfilled by block volumes through the CSI driver, and a volume follows its pod freely between nodes within one availability zone. Cross-zone moves are not transparent: they require a volume snapshot and a restore into the target zone, which the operator can automate but the platform does not do implicitly. StatefulSets spanning zones should therefore use per-zone storage classes and topology-aware scheduling.

**When does the autoscaler remove a node?**
A node becomes a removal candidate when its requested utilization stays below fifty percent for ten minutes and every pod on it can be rescheduled elsewhere without violating disruption budgets, affinity rules, or local storage constraints. Nodes annotated as no-scale-down are never touched, which protects singleton workloads that tolerate no eviction. The decision log of the autoscaler is exposed as events, so every removal or refusal can be traced to a rule.

**Can the cluster block risky workloads by policy?**
Built-in admission policies can reject privileged pods, host-network usage, and images without a passing vulnerability scan, evaluated before scheduling. Policies run in audit mode first, listing would-be violations, so enforcement can be turned on without breaking running teams.

**Are control-plane logs accessible?**
API server audit logs, scheduler decisions, and controller-manager events can be streamed per cluster to the logging service. Audit policy granularity is selectable per cluster from metadata-only up to request-and-response bodies for sensitive namespaces.

## Managed Databases

**Which database engines are offered?**
The managed service offers PostgreSQL and MySQL with community compatibility, plus Redis in both cache and persistent modes. Engine minor versions are patched automatically inside the weekly maintenance window, while major-version upgrades are always customer-initiated. Extensions are allow-listed per engine; PostGIS, pgvector, and the common observability extensions are pre-approved on PostgreSQL.

**How does high availability work for databases?**
A highly available instance runs a synchronous standby in a second availability zone, and automatic failover completes within thirty seconds of a confirmed primary failure. The connection endpoint is stable across failovers, so applications reconnect without configuration changes. Synchronous replication costs roughly five percent of write throughput, which the console shows explicitly before the option is enabled.

**What is the backup and restore story?**
Automated backups run nightly with a retention window configurable from seven to thirty-five days, and write-ahead logs are archived continuously, enabling point-in-time recovery to any second within the window. Restores always create a new instance, never overwrite the source, and a restored instance comes up in an isolated network by default until explicitly exposed. Manual snapshots are kept until deleted and survive instance deletion.

**Can I have read replicas in other regions?**
Up to five asynchronous read replicas are supported, including cross-region replicas for read locality and disaster recovery. Replication lag is exported as a first-class metric, and a replica can be promoted to a standalone primary in under a minute, which is the documented regional-failover path. Cross-region replication traffic is billed at the standard inter-region rate, with no markup.

**How are database credentials managed?**
The initial admin credential is generated at creation and stored in the secrets manager, never displayed in the console. Applications are expected to fetch credentials at startup through the secrets API or mounted secrets, and rotation re-issues the credential while keeping old sessions valid for a configurable grace period up to twenty-four hours. IAM-based authentication is available on PostgreSQL, removing static passwords entirely for supported clients.

**What maintenance windows apply?**
Each instance has a weekly thirty-minute maintenance window chosen at creation and changeable at any time, applied only when a patch actually needs it. Highly available instances patch the standby first, fail over, then patch the former primary, keeping the visible interruption under thirty seconds. Pending maintenance is announced at least five days ahead on the instance page and through a webhook.

**Is connection pooling built in?**
The built-in pooler multiplexes up to ten thousand client connections onto a bounded set of server connections, in transaction or session mode per database. Serverless and containerized applications should always connect through the pooler endpoint to avoid connection storms after deploys.

**How do I find slow queries without extra tooling?**
Query insights samples statements slower than one hundred milliseconds together with their execution plans and wait events. The insights page ranks statements by total time, and a plan-regression flag highlights queries whose plan changed after an engine patch.

## Serverless Functions

**What runtimes and limits apply to functions?**
Functions run Node.js, Python, and Go runtimes with a maximum execution time of fifteen minutes, memory from 128 MB to 10 GB, and a deployment package limit of 250 MB uncompressed. Concurrency defaults to one thousand parallel executions per project, raisable by request. Environment variables are encrypted at rest and capped at 4 KB total, which pushes larger configuration into the secrets manager.

**How does cold start behave and how do I reduce it?**
A cold start adds roughly one hundred to four hundred milliseconds for interpreted runtimes and under one hundred for Go, dominated by sandbox creation and dependency loading. Provisioned concurrency keeps a chosen number of sandboxes warm and removes cold starts entirely for that capacity, billed per warm-hour. Trimming the dependency tree matters more than code size: import time is the usual culprit in slow Python starts.

**How are functions triggered?**
Triggers include HTTP endpoints with optional IAM authorization, object-storage events on prefix and suffix filters, queue messages with batch sizes up to ten, and cron schedules with one-minute resolution. A single function can have multiple triggers, and each trigger carries its own retry policy. HTTP triggers support response streaming, which turns a function into a viable backend for server-sent events.

**What happens when a function fails?**
Synchronous invocations return the error to the caller and are never retried by the platform, while asynchronous invocations are retried twice with exponential backoff. After the final failure the event goes to the project's dead-letter queue if one is configured, with the full payload and the error chain attached. The failure-rate metric distinguishes handled errors thrown by code from sandbox crashes, which simplifies triage.

**Can functions reach my private network?**
Attaching a function to a VPC connector places its sandboxes inside a chosen subnet, granting access to private databases and internal services. The attachment adds about fifty milliseconds to cold starts and pins egress through the subnet's routing, including any NAT gateway. Without a connector, functions have internet egress only and cannot reach private address space at all.

**How is function usage billed?**
Billing combines invocation count with gigabyte-seconds of memory-time, measured in one-millisecond increments with no minimum duration. Provisioned concurrency is billed per warm-hour regardless of traffic, and VPC connectors add a flat hourly charge per connector, not per function. The free monthly allowance of 100,000 invocations and 400,000 gigabyte-seconds applies per organization, not per project, and never converts to paid automatically.

**Can I roll out a new function version gradually?**
Aliases can split traffic between two versions by percentage, and the split can shift automatically on a schedule or halt on an alarm signal. Rollback is a metadata change on the alias, taking effect for new invocations within seconds.

**How do I test functions locally?**
The emulator container image reproduces the production sandbox, including memory limits, the credentials endpoint, and trigger payload shapes. Contract tests against the emulator in CI catch the classic drift between local behavior and the deployed runtime.

## Monitoring and Logging

**What metrics are collected by default?**
Every resource exports its core metrics at one-minute resolution with ninety days of retention at no charge: CPU, memory, disk, and network for compute, request counts and latencies for balancers, and engine health for databases. Custom metrics can be pushed through the API at ten-second resolution and are billed per active series per month. Percentile aggregations are computed at ingestion, so p95 and p99 queries stay fast on any time range.

**How do alerts avoid flapping?**
Alert rules combine a threshold with a mandatory hold duration, so a spike shorter than the hold never fires, and resolution requires the condition to stay clear for the same duration. Notification channels include email, webhooks, and the major paging services, with per-channel severity filters. A silencing window can mute any rule set during planned maintenance, and silences require an expiry, so nothing stays muted forever by accident.

**Can I trace a request across services?**
Distributed tracing accepts W3C trace context and OpenTelemetry exports, stitching spans from functions, Kubernetes workloads, and balancers into one trace view. Sampling is head-based at a configurable rate with an always-sample override for traces that touch an error. Trace retention is fourteen days, and any trace can be exported as JSON for offline analysis or attachment to a ticket.

**How long are audit logs kept and can they be exported?**
Every control-plane API call lands in the audit trail with actor, source IP, and full request payload, immutable and retained for 400 days. A streaming export can copy audit records to a bucket or an external SIEM within one minute of the event, and the export itself is recorded in the trail. Read access to the audit trail is a separate permission, so security teams can grant visibility without granting any operational power.

**What does the logging query language support?**
The query language filters by any indexed field, supports full-text search over the message body, and computes aggregations, percentiles, and time-bucketed histograms directly in the query. Saved queries can back dashboards and alerts alike, so an investigation query becomes a monitor with one click. Query cost is billed by scanned volume, and the editor shows the scan estimate before execution to keep exploratory costs predictable.

**Can logs and metrics from outside Nimbus be ingested?**
The ingestion endpoints accept OpenTelemetry, Prometheus remote-write, and syslog over TLS from any source with a valid ingestion key, so hybrid estates can converge on one observability stack. Ingestion keys are scoped per source and revocable individually, and rate limits are configured per key to protect the pipeline from a misbehaving sender. External data is billed at the same rates as platform-generated data with no ingress surcharge.

**Are synthetic uptime checks included?**
Synthetic probes run from eight geographic locations at one-minute intervals against HTTP, TCP, or ICMP targets, with certificate-expiry assertions built in. Probe failures create incidents with per-location evidence, separating a regional network issue from a global outage.

**Can I share a dashboard with someone outside the organization?**
Dashboards can be shared read-only through a signed link with a mandatory expiry of at most thirty days. Shared views render from snapshots of the queries, never live credentials, so revoking the link is the only cleanup required.

## Backups and Disaster Recovery

**What does the backup service cover?**
The backup service provides scheduled, policy-driven backups for block volumes, databases, and Kubernetes persistent volumes under one retention policy language. Policies express schedules, retention tiers, and target vaults, and every policy application is recorded so compliance reviews can prove coverage. Resources tagged as backup-exempt are listed in a weekly exception report to keep silent gaps from forming.

**What is a backup vault and why does it matter?**
A vault is an isolated storage boundary for backup data with its own access policy, encryption keys, and optional immutability lock. A compromised project credential cannot delete vault contents when the lock is active, because lock removal requires a time-delayed, dual-approval procedure. Vaults can live in a different region from the protected resources, which is the recommended posture for regional disaster recovery.

**How does immutability protect against ransomware?**
With the immutability lock enabled, backup points cannot be deleted or shortened below the configured retention by anyone, including organization owners, until the retention elapses. Lock changes take effect after a seventy-two-hour cooling period announced to all owners, which defeats smash-and-grab credential abuse. Restore operations remain instant and unrestricted, so the lock protects the data without slowing recovery.

**What recovery objectives can the platform meet?**
Volume restores complete at roughly one terabyte per hour into the same zone, database point-in-time recovery typically lands within minutes for instances under one terabyte, and cross-region replica promotion completes in under a minute once initiated. Achievable recovery-point objectives range from seconds, with continuous log archiving, to twenty-four hours with plain nightly backups. The disaster-recovery planner simulates a chosen scenario and reports the projected recovery time per resource.

**Can I test restores without touching production?**
Restore rehearsals create resources in an isolated network with no route to production, from any chosen backup point. A rehearsal report captures restore duration, data verification results, and the exact commands used, forming an auditable record for compliance frameworks that require periodic testing. Rehearsal resources are billed at normal rates but are auto-deleted after a configurable lifetime, two hours by default.

**How is backup usage billed?**
Backup storage is billed per GB-month of unique compressed blocks in the vault, and restores within the same region are free of data charges. Cross-region restore traffic is billed at the standard inter-region rate, and immutability-locked data is billed identically to unlocked data. The cost page attributes backup spend per protected resource, which makes it easy to spot a runaway retention policy.

**Can Kubernetes backups be application-consistent?**
Pre- and post-backup hooks run inside annotated pods, letting databases flush and quiesce before the volume snapshot is taken. Hook failures mark the backup point as crash-consistent-only rather than silently pretending consistency.

**What is a legal hold and how does it interact with retention?**
A legal hold pins selected backup points indefinitely, overriding every retention and lifecycle rule until the hold is explicitly released. Holds require a case reference, appear in the compliance report, and their creation and release are both recorded in the audit trail.

## Security and Compliance

**Which certifications does the platform hold?**
The platform is certified for ISO 27001, SOC 2 Type II, and PCI DSS Level 1, and offers a signable data processing agreement with EU standard contractual clauses. Certification reports and the current bridge letters are available for download under NDA in the trust center. The scope statement of each certification names the exact services covered, and preview services are explicitly out of scope until listed.

**How do I report a security vulnerability?**
Email security@nimbus.example with reproduction steps; the security team acknowledges within one business day and assigns a severity within three. The coordinated disclosure window is ninety days, extendable by mutual agreement when a fix requires ecosystem coordination. A public hall of fame credits reporters who consent, and monetary bounties apply to the categories listed in the program page, paid within thirty days of triage.

**How are secrets stored and delivered to workloads?**
The secrets manager stores versioned secrets encrypted under a hardware security module, and every read is logged with the requesting identity. Secrets can be mounted into instances, containers, and functions at boot without ever appearing in configuration files or environment listings. Rotation hooks let a secret owner re-issue credentials and propagate them to consumers with zero-downtime, following a documented two-phase pattern.

**What happens to data on failed hardware?**
Failed drives are securely erased where possible and physically destroyed otherwise, following NIST 800-88, before any component leaves the datacenter, with a certificate of destruction retained per serial number. Multi-replica storage rebuilds automatically onto healthy hardware, and rebuild progress is visible as a metric. Customer data never leaves the region on a physical device under any operational procedure.

**Is penetration testing by customers allowed?**
Customers may test their own workloads at any time without prior approval, except for denial-of-service scenarios, which always require a scheduled engagement approved by the security team. Testing that strays onto platform infrastructure or other tenants is prohibited and technically constrained by isolation. A courtesy notification before large tests avoids automated abuse responses such as rate limiting or address blocking.

**How fast are hypervisors patched after a critical CVE?**
Critical hypervisor vulnerabilities are patched fleet-wide within seventy-two hours using live migration, with no customer-visible downtime in the general-purpose fleet. The metal series, which cannot live-migrate, receives coordinated reboot windows announced at least five days ahead unless active exploitation forces emergency action. A post-patch bulletin lists affected components and the exact remediation timeline per region.

**How strong is isolation between tenants?**
Projects are isolated by hardware-virtualized boundaries, dedicated per-tenant encryption contexts, and network segmentation enforced below the hypervisor. Side-channel mitigations follow vendor guidance, and the isolation architecture is described in a whitepaper available under NDA.

**Can API access be restricted by network origin?**
An organization policy can restrict control-plane API access to declared CIDR ranges, with per-service exceptions for automation running outside them. Requests from outside the allowlist are rejected before authentication and logged with the offending source address.

## Support Plans and Migration

**What support plans exist and what do they cost?**
The Basic plan is free and covers billing and account issues with a two-business-day response. Developer, at ninety-nine euros per month, adds technical support with a one-business-day response on all severities. Business, at three percent of monthly spend with a five-hundred-euro floor, brings one-hour response on production-down incidents and access to the architecture review service. Enterprise adds a named technical account manager, fifteen-minute response on critical incidents, and quarterly operational reviews.

**How do severity levels work on tickets?**
Severity one means a production system is down or severely degraded with no workaround, and on Business and Enterprise it triggers a phone bridge with an on-call engineer. Severity is set by the customer at creation, but support may reclassify with a written justification, and reclassification restarts the response clock. Response-time commitments are contractual on paid plans, and monthly reports show attainment per severity.

**Is there help for migrating from other providers?**
The migration service assesses source inventory, plans waves, and executes agent-based replication for machines and databases with cutover windows measured in minutes. Migration tooling is free; customers pay only for the target resources and any inter-cloud transfer charged by the source provider. Organizations moving more than fifty terabytes qualify for the data-onboarding program, which ships encrypted transfer appliances and credits inbound transfer costs.

**Can I get architectural review before going to production?**
The architecture review service, included from the Business plan, examines a workload against the reliability, security, cost, and operations pillars and produces a prioritized findings report. Reviews are conversation-based with a solutions architect and typically take two sessions of ninety minutes. A follow-up session six weeks later checks remediation progress, and unresolved critical findings are flagged in the account's risk register.

**How do I escalate when a ticket stalls?**
Every ticket shows a visible escalate action after the first response commitment lapses, which pages the duty manager of the support organization. Enterprise customers can additionally page their technical account manager at any time for coordination. Escalations are reviewed weekly by support leadership, and systemic causes feed the platform's public post-incident reports when they stem from an outage.

**Where do I find scheduled maintenance announcements?**
The status page lists live incidents and scheduled maintenance per region and per service, with subscription by webhook, email, and RSS. Maintenance that can affect running workloads is announced at least five days ahead, and emergency maintenance carries a reason and a blast-radius statement. Historical uptime per service is published monthly, twelve months deep, computed from the same probes that drive the SLA.

**Is there a free community support channel?**
The community forum is public, searchable, and staffed by platform engineers on a best-effort basis, with verified answers marked distinctly. Bug reports confirmed on the forum are converted into tracked issues, and the reporter receives status updates automatically.

**Does Enterprise include training?**
Enterprise includes twenty training credits per year, redeemable for instructor-led courses or certification exams. Credits are organization-wide, expire at contract renewal, and unused balances appear in the quarterly operational review alongside a consumption plan.

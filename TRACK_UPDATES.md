# Track B Updates

## Clarification on output content and format

1. For questions that require the output of topology links, the port numbers in the answer must be written in the following format: the local/peer port number is written based on the interface name in the command output of the `display current-configuration` command. If the `display current-configuration` command is unavailable, the interface name in the command output of the `display interface brief` command is used. If the interface information is followed by other information, such as the interface bandwidth or rate, this information is not retained.
2. For questions that require the output of paths, all nodes, including nodes on L2 paths, must be included in the answer. If there are multiple paths, each path must be output on a separate line.
3. For questions that require the identification of fault causes, you must select the most specific and closest fault cause.

## Network device list

To help participants better answer the questions in Track B, the following device names are involved in the network:

`AGG_SW_01`, `AGG_SW_02`, `AGG_SW_03`, `AGG_SW_04`, `BJHQ_CSR1000V_GW_01`, `BaiduWebServer01`, `ChinaUnicom_SW`, `Core_SW_01`, `Core_SW_02`, `EMPLOYEE_WIFI_CLIENT01`, `EMPLOYEE_WIFI_CLIENT02`, `EMPLOYEE_WIFI_CLIENT03`, `FW_01`, `FW_02`, `GUEST_WIFI_CLIENT01`, `GUEST_WIFI_CLIENT02`, `GUEST_WIFI_CLIENT03`, `GoogleWebServer01`, `HQ-DHCP-Server`, `HQ_DNS_Server_01`, `HQ_FIN_Client01`, `HQ_FIN_PC01`, `HQ_FTP_Server_01`, `HQ_HR_AP01`, `HQ_HR_PC01`, `HQ_HTTP_Server_01`, `HQ_MKT_AP01`, `HQ_MKT_Client01`, `HQ_MKT_PC01`, `HQ_PROC_AP01`, `HQ_PROC_PC01`, `Internet_PC01`, `Outside_FTP_Client01`, `PE1`, `PE2`, `PE3`, `SH_AR`, `SH_Core`, `SH_FAC_PC01`, `SH_SAL_PC01`, `SH_STO_PC01`, `SW-DMZ-ACC-01`, `SZ_AR`, `SZ_Core`, `SZ_Server_Cluster1`, `SZ_Server_Cluster2`, `SZ_Server_Cluster3`

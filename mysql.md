
fdisk /dev/sdb
mkfs -t ext4 /dev/sdb1
mount /dev/sdb1 /apps

https://www.cnblogs.com/brianzhu/p/8575243.html


修改location
vi /etc/my.cnf

set password for 'root'@'localhost'=password('Had00p!!'); 
systemctl stop firewalld


mysql x 
#https://dev.mysql.com/doc/refman/5.7/en/document-store-setting-up.html
mysqlsh -u root -h localhost --classic --dba enableXProtocol
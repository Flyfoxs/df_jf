
fdisk /dev/sdb
mkfs -t ext4 /dev/sdb1
mount /dev/sdb1 /apps

https://www.cnblogs.com/brianzhu/p/8575243.html


修改location
vi /etc/my.cnf

set password for 'root'@'localhost'=password('Had00p!!'); 
systemctl stop firewalld
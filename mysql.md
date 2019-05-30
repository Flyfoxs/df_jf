


fdisk /dev/sdb  ## n, p, w

mkfs -t ext4 /dev/sdb1 

mount /dev/sdb1 /apps

vi /etc/fstab
###
/dev/sdb1 /apps   ext4  defaults       0  0

https://www.cnblogs.com/brianzhu/p/8575243.html


修改location
vi /etc/my.cnf

set password for 'root'@'localhost'=password('Had00p!!'); 
systemctl stop firewalld


mysql x 
#https://dev.mysql.com/doc/refman/5.7/en/document-store-setting-up.html
mysqlsh -u root -h localhost --classic --dba enableXProtocol

sudo apt-get update && sudo apt-get upgrade
sudo apt-get install -y tlp tlp-rdw
# Once installed, run the command below to start it:

sudo tlp start
sudo apt-get install curl
sudo apt-get install -y git
sudo apt-get install -y build-essential libbz2-dev libssl-dev libreadline-dev \
                        libsqlite3-dev tk-dev

# optional scientific package headers (for Numpy, Matplotlib, SciPy, etc.)
sudo apt-get install -y libpng-dev libfreetype6-dev
curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash

echo export PATH="~/.pyenv/bin:$PATH" >> ~/.bashrc
echo eval "$(pyenv init -)"  >> ~/.bashrc
echo eval "$(pyenv virtualenv-init -)" >>~/.bashrc
source ~/.bashrc

pyenv install 3.6.0
pyenv virtualenv 3.6.0 general
pyenv global general

echo export PATH="~/.pyenv/bin:$PATH" >> ~/.bashrc
echo eval "$(pyenv init -)"  >> ~/.bashrc
echo eval "$(pyenv virtualenv-init -)" >>~/.bashrc
source ~/.bashrc



pip install python-telegram-bot psycopg2 sqlalchemy  jupyter jupyter_contrib_nbextensions autopep8 pandas
sudo apt-get install postgresql postgresql-contrib
sudo -u postgres createuser eamag
sudo -u postgres createdb eamag
sudo locale-gen ru_RU.UTF-8
# sudo dpkg-reconfigure locales

#$ sudo -u postgres psql
#psql=# alter user eamag with encrypted password 'qweqwe';
#psql=# grant all privileges on database eamag to eamag ;
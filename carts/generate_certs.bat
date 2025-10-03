@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo   Certificate Generation Tool (Windows)
echo ==========================================

echo Creating certs directory...
if not exist "certs" mkdir "certs"
cd /d "certs"

echo.
echo Generating CA root certificate...
openssl genrsa -out ca.key 2048
openssl req -x509 -new -nodes -key ca.key -days 3650 -out ca.crt -subj "/C=CN/CN=vproxy-ca"

echo.
echo Generating api.openai.com domain certificate...
openssl genrsa -out api.openai.com.key 2048
openssl req -new -key api.openai.com.key -out api.openai.com.csr -subj "/C=CN/CN=api.openai.com"
openssl x509 -req -in api.openai.com.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out api.openai.com.crt -days 3650

echo.
echo Certificate generation completed successfully!
echo.
echo Generated files:
echo - CA private key: ca.key
echo - CA certificate: ca.crt
echo - Domain private key: api.openai.com.key
echo - Certificate signing request: api.openai.com.csr
echo - Domain certificate: api.openai.com.crt
echo.

pause
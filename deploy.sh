#!/bin/bash
mvn -Dmaven.test.skip -Drat.skip=true -Drat.numUnapprovedLicenses=1000 package && cp guacamole/target/guacamole-1.4.0.war ../nv-apps/guacamole/guac-xfce-sidecar/

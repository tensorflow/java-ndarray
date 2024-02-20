# Releasing NdArray Java Library

The
[NdArray Java Library](https://github.com/tensorflow/java-ndarray) is available on Maven Central and JCenter
through artifacts uploaded to
[OSS Sonatype](https://oss.sonatype.org/content/repositories/releases/org/tensorflow/). This
document describes the process of updating the release artifacts. It does _not_ describe how to use
the artifacts, for which the reader is referred to the
[NdArray Java Library installation instructions](https://github.com/tensorflow/java-ndarray/blob/main/README.md).

## Release process overview

NdArray Java Library must be conducted locally in a [Docker](https://www.docker.com) container for a hermetic release process.

It is important to note that any change pushed to a release branch (i.e. a branch prefixed
by `r`) will start a new release workflow. Therefore, these changes should always increment the
version number.

### Pre-requisites

-   `docker`
-   An account at [oss.sonatype.org](https://oss.sonatype.org/), that has
    permissions to update artifacts in the `org.tensorflow` group. If your
    account does not have permissions, then you'll need to ask someone who does
    to [file a ticket](https://issues.sonatype.org/) to add to the permissions
    ([sample ticket](https://issues.sonatype.org/browse/MVNCENTRAL-1637)).
-   A GPG signing key, required
    [to sign the release artifacts](http://central.sonatype.org/pages/apache-maven.html#gpg-signed-components).

### Preparing a release

#### Major or minor release

1.  Get a clean version of the source code by cloning the
    [NdArray Java Library GitHub repository](https://github.com/tensorflow/java-ndarray)
    ```
    git clone https://github.com/tensorflow/java-ndarray
    ```
2.  Create a new branch for the release named `r<MajorVersion>.<MinorVersion>`
    ```
    git checkout -b r1.0
    ```
3.  Update the version of the Maven artifacts to the full version of the release
    ```
    mvn versions:set -DnewVersion=1.0.0
    ```
4.  Update the NdArray Java Library version to reflect the new release at the following locations:
    - https://github.com/tensorflow/java-ndarray/blob/main/README.md#introduction

5.  Commit the changes and push the branch to the GitHub repository
    ```
    git add .
    git commit -m "Releasing 1.0.0"
    git push --set-upstream origin r1.0
    ```

#### Patch release

1.  Get a clean version of the source code by cloning the
    [NdArray Java Library GitHub repository](https://github.com/tensorflow/java-ndarray)
    ```
    git clone https://github.com/tensorflow/java-ndarray
    ```
2.  Switch to the release branch of the version to patch
    ```
    git checkout r1.0
    ```
3.  Patch the code with your changes. For example, changes could be merged from another branch you
    were working on or be applied directly to this branch when the required changes are minimal.

4.  Update the version of the Maven artifacts to the full version of the release
    ```
    mvn versions:set -DnewVersion=1.0.1
    ```
5.  Update the NdArray Java Library version to reflect the new release at the following locations:
    - https://github.com/tensorflow/java-ndarray/blob/main/README.md#introduction

6.  Commit the changes and push the branch to the GitHub repository
    ```
    git add .
    git commit -m "Releasing 1.0.1"
    git push
    ```

### Performing the release

1.  At the root of your repository copy, create a Maven settings.xml file with your OSSRH credentials and
    your GPG key passphrase:
    ```sh
    SONATYPE_USERNAME="your_sonatype.org_username_here"
    SONATYPE_PASSWORD="your_sonatype.org_password_here"
    GPG_PASSPHRASE="your_gpg_passphrase_here"
    cat > settings.xml <<EOF
    <settings>
      <servers>
        <server>
          <id>ossrh</id>
          <username>${SONATYPE_USERNAME}</username>
          <password>${SONATYPE_PASSWORD}</password>
        </server>
        <server>
          <id>ossrh-staging</id>
          <username>${SONATYPE_USERNAME}</username>
          <password>${SONATYPE_PASSWORD}</password>
        </server>
      </servers>
      <profiles>
        <profile>
          <activation>
            <activeByDefault>true</activeByDefault>
          </activation>
          <properties>
            <gpg.executable>gpg2</gpg.executable>
            <gpg.passphrase>${GPG_PASSPHRASE}</gpg.passphrase>
          </properties>
        </profile>
      <profiles>
    </settings>
    EOF
    ```
2.  Execute the `release.sh` script. This will sign and deploy artifacts on OSS Sonatype.

    On success, the released artifacts are uploaded to the private staging repository in OSS Sonatype (check at the Maven
    build output to know the exact location of the staging repository). After inspecting the artifacts in OSS Sonatype, you 
    should release or drop them.
    
    Visit https://oss.sonatype.org/#stagingRepositories, find the `orgtensorflow-*`
    of your release and click `Close` and `Release` to finalize the release. You always have the option
    to `Drop` it to abort and restart if something went wrong.

3.  Go to GitHub and create a release tag on the release branch with a summary of what the version includes.

Some things of note:
    - For details, look at the [Sonatype guide](http://central.sonatype.org/pages/releasing-the-deployment.html).
    - Syncing with [Maven Central](http://repo1.maven.org/maven2/org/tensorflow/)
      can take 10 minutes to 2 hours (as per the [OSSRH guide](http://central.sonatype.org/pages/ossrh-guide.html#releasing-to-central)).

### Finishing a release

#### Major or minor release

1. Checkout the main branch and merge back changes from the released branch
   ```
   git checkout main
   git merge r1.0
   ```
2. In your local copy, checkout the main branch and increase the next snapshot version.
   ```
   mvn versions:set -DnewVersion=1.1.0-SNAPSHOT
   ```
3. Update the NdArray Java Library version to reflect the new release at the following locations:
    - https://github.com/tensorflow/java-ndarray/blob/main/README.md#introduction

4. Commit your changes and push the main branch to the GitHub repository
   ```
   git add .
   git commit -m "Increase version for next iteration"
   git push
   ```

#### Patch release

1. Checkout the main branch and merge back changes from the released branch, preserving current snapshot version
   ```
   git checkout main
   git merge r1.0
   ```
2. Commit the main and push the main branch to the GitHub repository
   ```
   git add .
   git commit -m "Merge release 1.0.1"
   git push
   ```

## References

-   [Sonatype guide](http://central.sonatype.org/pages/ossrh-guide.html) for
    hosting releases.
-   [Ticket that created the `org/tensorflow` configuration](https://issues.sonatype.org/browse/OSSRH-28072) on OSSRH.

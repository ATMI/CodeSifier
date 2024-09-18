FROM ubuntu:16.04

COPY scripts/android-base-apt-get.sh /scripts/
RUN sh /scripts/android-base-apt-get.sh

COPY scripts/android-ndk.sh /scripts/
RUN . /scripts/android-ndk.sh && \
    download_and_make_toolchain android-ndk-r15c-linux-x86_64.zip x86_64 21

ENV PATH=$PATH:/android/ndk/x86_64-21/bin

ENV DEP_Z_ROOT=/android/ndk/x86_64-21/sysroot/usr/

ENV HOSTS=x86_64-linux-android

ENV RUST_CONFIGURE_ARGS \
      --x86_64-linux-android-ndk=/android/ndk/x86_64-21 \
      --disable-rpath \
      --enable-extended \
      --enable-cargo-openssl-static

ENV SCRIPT python3 ../x.py dist --target $HOSTS --host $HOSTS

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

# injected fil files 
I found out that h5 works better than fil generally, so I started exporting in
that format instead.

Files with the long name ending in `.dat` are seticore's outputs, while files
`turboSETI.[adj].dat` are turboSETI outputs. 

the adjective in `[adj]` describes how I modified the signal:
- **clean** means I didn't at all, that's the output from the original file
- _(nothing)_ means I added a 1.1 Hz/s drift shift constant signal at 25 SNR,
  starting at the frequency of 18392809557.35594 Hz
- **strong** same as _(nothing)_, but at about 50 SNR
- **mighty** means I added a 5 Hz/s constant signal at 3000 SNR, starting at the
  frequency of 18392808998.562397 Hz

## results
at the moment only turboSETI can detect something, and only the **mighty**
signal.

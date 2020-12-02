## FIXED
- The forward method of the model was not actually being implemented, the try-catch statement wasn't working for some reason so I've turned apex off and commented out the stupid try-catch statement.

## Error logs:

**VS-Code error**:
Exception has occurred: NotImplementedError

**Error in trainer.train() variables**:
RuntimeError('imag is not implemented for tensors with non-complex dtypes.')
RuntimeError('real is not implemented for tensors with non-complex dtypes.')

- This happened when running trainer.train() in the DPPM code fork
- This shows up in the debug window inside the data

The data in the Trainer object contains this error, doesn't give any more details than that (helpfully...)

## Attempted fixes
- Turning fp16 off
- checking github of original package
- [here](https://github.com/fastai/fastai2/issues/395) the issue was raised but resolved on the next pull request
- updating all packages
- Changing filepath to pick up right files (it's picking up the files I think!)
- Elimintation all error reports
  - Updated pylint settings using [this fix](https://github.com/pytorch/pytorch/issues/701)

## Next steps
- [x] Self code walk, see if I can find the error - want to make sure it's coming from trainer before I rebuild it (won't be able to test otherwise)
  - [ ] Draw diagram as you go
- [ ] Build my own trainer if I can't find the error
- [ ] Failing that, e-mail chris and ask who he'd recommend I speak to, fall back to reading when corresponding



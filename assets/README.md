# assets

`kvpack-social-card.png` is the 1200×630 social preview image. It is
generated deterministically — do not edit it by hand:

```sh
python3 scripts/generate_social_card.py           # regenerate
python3 scripts/generate_social_card.py --check   # verify committed copy
```

The GitHub social preview is configured manually: repo → Settings → General
→ Social preview → upload this file.

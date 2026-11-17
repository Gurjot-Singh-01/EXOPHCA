import tempfile
import pathlib
import shutil
import subprocess

src = pathlib.Path("data/reference/hzcalc.py")
with tempfile.TemporaryDirectory() as tmp:
    tmp_path = pathlib.Path(tmp)
    script_copy = tmp_path / "hzcalc.py"
    shutil.copy(src, script_copy)
    content = script_copy.read_text()

    # Replace print statements with Python 3 print() calls
    content = content.replace("print '************************************************************'", "print('************************************************************')")
    content = content.replace("print ''", "print('')")
    content = content.replace("print 'The HZ coefficients are printed in HZ_coefficients.dat file.'", "print('The HZ coefficients are printed in HZ_coefficients.dat file.')")
    content = content.replace("print 'HZs for stars with 2600 K <= Teff <=7200 K is in HZs.dat file.'", "print('HZs for stars with 2600 K <= Teff <=7200 K is in HZs.dat file.')")

    script_copy.write_text(content)
    subprocess.run(["python", "-I", str(script_copy)], cwd=tmp, check=True)

    hzs_file = tmp_path / "HZs.dat"
    lines = hzs_file.read_text().splitlines()

    target = pathlib.Path("data/reference/hz_fluxes_oracle.csv")
    out_lines = ["teff,recent_venus,runaway_greenhouse,maximum_greenhouse,early_mars,runaway_5me,runaway_0p1me"]
    for l in lines[2:]:
        parts = l.split()
        if len(parts) >= 7:
            out_lines.append(",".join(parts[:7]))
    target.write_text("\n".join(out_lines) + "\n")
    print(f"Successfully generated {target}")

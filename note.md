cli.BalanceStand(         cli.Init(
cli.BodyHeight(           cli.Move(
cli.ContinuousGait(       cli.Pose(
cli.Damp(                 cli.RecoveryStand(
cli.Dance1(               cli.RiseSit(
cli.Dance2(               cli.Scrape(
cli.EconomicGait(         cli.SetTimeout(
cli.Euler(                cli.Sit(
cli.FootRaiseHeight(      cli.SpeedLevel(
cli.FrontFlip(            cli.StandDown(
cli.FrontJump(            cli.StandUp(
cli.FrontPounce(          cli.StopMove(
cli.GetApiVersion(        cli.Stretch(
cli.GetFootRaiseHeight(   cli.SwitchGait(
cli.GetLeaseId(           cli.SwitchJoystick(
cli.GetServerApiVersion(  cli.TrajectoryFollow(
cli.GetSpeedLevel(        cli.Trigger(
cli.GetState(             cli.WaitLeaseApplied(
cli.Heart(                cli.Wallow(
cli.Hello(                cli.WiggleHips(
-----------------
# Value shielding
Good runs:
With action_smoothing=0.5
```bash
python3 test_mjlab_value_shielding_numpad.py --epsilon -0.34 --ctrl_step 35000000 --kp 100,100,200 --lx_enter_stance -0.1
```
```bash
python3 test_mjlab_value_shielding_numpad.py --epsilon -0.33 --ctrl_step 45800000 --kp 100,100,200 --lx_enter_stance -0.1
```
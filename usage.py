from speaker import Speaker
mySpeaker = Speaker(azi=0, elev=0, track="track.mp3", order=1)
mySpeaker.spin_horizontal(revolutions=10, speed=2, clockwise=True, sofa_path="sofa.sofa")

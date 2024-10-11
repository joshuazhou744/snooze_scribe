import React from 'react'
import { Link } from 'react-router-dom'
import "./Manual.css"

export default function Manual() {
  return (
    <div className='container'>
        <div className="header">
            <Link to={"/"}>
                <button className='home-button'>Home</button>
            </Link>
        </div>

        <div className="quickstart">
            <h2>User Guide</h2>
            <ol className='quickstart-list'>
              <li>Follow the Calibration Instructions on the homepage to set an appropriate energy threshold</li>
              <li>Place your mobile device face down on a bedside table or somewhere near your bed</li>
              <li>Plug a charger into your mobile device and ensure it is charging before sleeping</li>
              <li>Begin recording and have a good night</li>
            </ol>
        </div>

        <div className="additional">
            <h2>Additional Information</h2>
            <ul className='additional-list'>
              <li>The audio playback feature will NOT work on IOS mobile browsers due to Webkit API restrictions</li>
              <li>Please use any other device to view and play your audio files; A fix is in progress</li>
              <li>Please refer to the Github Repository listed below for version history and future updates</li>
              <li>Source code: <a href="https://github.com/joshuazhou744/snooze_scribe" target='_blank'>https://github.com/joshuazhou744/snooze_scribe</a></li>
              <li>Developer Contact: joshuazhou744@gmail.com</li>
            </ul>
        </div>
    </div>
  )
}

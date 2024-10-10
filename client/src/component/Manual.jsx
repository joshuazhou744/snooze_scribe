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
              <li>Follow Calibration Instructions on homepage to set an appropriate energy threshold</li>
              <li>Place mobile device face down on a bedside table or somewhere near your bed</li>
              <li>Plug a charger into the mobile device and ensure it is charging before sleeping</li>
              <li>Begin recording and good night</li>
            </ol>
        </div>

        <div className="additional">
            <h2>Additional Information</h2>
            <ul className='additional-list'>
              <li>Audio playback will NOT work on IOS mobile browsers due to Webkit API restrictions</li>
              <li>Your audio data is NOT being sold or used for nefarious purposes</li>
              <li>Source code: https://github.com/joshuazhou744/snooze_scribe</li>
              <li>Developer Contact: joshuazhou744@gmail.com</li>
            </ul>
        </div>
    </div>
  )
}

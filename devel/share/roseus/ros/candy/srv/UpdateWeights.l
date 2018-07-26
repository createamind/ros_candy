;; Auto-generated. Do not edit!


(when (boundp 'candy::UpdateWeights)
  (if (not (find-package "CANDY"))
    (make-package "CANDY"))
  (shadow 'UpdateWeights (find-package "CANDY")))
(unless (find-package "CANDY::UPDATEWEIGHTS")
  (make-package "CANDY::UPDATEWEIGHTS"))
(unless (find-package "CANDY::UPDATEWEIGHTSREQUEST")
  (make-package "CANDY::UPDATEWEIGHTSREQUEST"))
(unless (find-package "CANDY::UPDATEWEIGHTSRESPONSE")
  (make-package "CANDY::UPDATEWEIGHTSRESPONSE"))

(in-package "ROS")





(defclass candy::UpdateWeightsRequest
  :super ros::object
  :slots (_a ))

(defmethod candy::UpdateWeightsRequest
  (:init
   (&key
    ((:a __a) "")
    )
   (send-super :init)
   (setq _a (string __a))
   self)
  (:a
   (&optional __a)
   (if __a (setq _a __a)) _a)
  (:serialization-length
   ()
   (+
    ;; string _a
    4 (length _a)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _a
       (write-long (length _a) s) (princ _a s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _a
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _a (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass candy::UpdateWeightsResponse
  :super ros::object
  :slots (_b ))

(defmethod candy::UpdateWeightsResponse
  (:init
   (&key
    ((:b __b) "")
    )
   (send-super :init)
   (setq _b (string __b))
   self)
  (:b
   (&optional __b)
   (if __b (setq _b __b)) _b)
  (:serialization-length
   ()
   (+
    ;; string _b
    4 (length _b)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _b
       (write-long (length _b) s) (princ _b s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _b
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _b (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass candy::UpdateWeights
  :super ros::object
  :slots ())

(setf (get candy::UpdateWeights :md5sum-) "945e963769938e4ddc3288e80fdfddf4")
(setf (get candy::UpdateWeights :datatype-) "candy/UpdateWeights")
(setf (get candy::UpdateWeights :request) candy::UpdateWeightsRequest)
(setf (get candy::UpdateWeights :response) candy::UpdateWeightsResponse)

(defmethod candy::UpdateWeightsRequest
  (:response () (instance candy::UpdateWeightsResponse :init)))

(setf (get candy::UpdateWeightsRequest :md5sum-) "945e963769938e4ddc3288e80fdfddf4")
(setf (get candy::UpdateWeightsRequest :datatype-) "candy/UpdateWeightsRequest")
(setf (get candy::UpdateWeightsRequest :definition-)
      "string a
---
string b
")

(setf (get candy::UpdateWeightsResponse :md5sum-) "945e963769938e4ddc3288e80fdfddf4")
(setf (get candy::UpdateWeightsResponse :datatype-) "candy/UpdateWeightsResponse")
(setf (get candy::UpdateWeightsResponse :definition-)
      "string a
---
string b
")



(provide :candy/UpdateWeights "945e963769938e4ddc3288e80fdfddf4")



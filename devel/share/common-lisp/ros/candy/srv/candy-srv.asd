
(cl:in-package :asdf)

(defsystem "candy-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "Step" :depends-on ("_package_Step"))
    (:file "_package_Step" :depends-on ("_package"))
    (:file "UpdateWeights" :depends-on ("_package_UpdateWeights"))
    (:file "_package_UpdateWeights" :depends-on ("_package"))
    (:file "Value" :depends-on ("_package_Value"))
    (:file "_package_Value" :depends-on ("_package"))
  ))
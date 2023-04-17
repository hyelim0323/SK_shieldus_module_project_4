from flask import Blueprint

# 인증 관련 서비스 -> 모든 URL은 ~/auth/~
bp_auth = Blueprint('auth_bp', 
                    __name__,  
                    url_prefix='/auth',             
                    template_folder='../templates/auth', 
                    static_folder='../static'       
                    )

# 채팅 관련 서비스 -> 모든 URL은 ~/chat/~
bp_chat = Blueprint('chat_bp', 
                    __name__,  
                    url_prefix='/chat',             
                    template_folder='../templates/chat', 
                    static_folder='../static'       
                    )
